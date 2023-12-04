import copy
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import numpy
import torch

import torch.nn.functional as F

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("deepsparse")
class DeepSparseLM(LM):
    # Default max sequence length setting for when no `max_length` is provided
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.
        """
        super().__init__()

        self._batch_size = int(batch_size)
        self._max_length = max_length or self._DEFAULT_MAX_LENGTH
        self._max_gen_toks = max_gen_toks

        import deepsparse

        # Initialize new model and tokenizer instances
        self.model = deepsparse.TextGeneration(
            model_path=pretrained,
            sequence_length=self._max_length,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer

        self.vocab_size = self.tokenizer.vocab_size

    def _loglikelihood_tokens(self, requests):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        chunks = utils.chunks(re_ord.get_reordered(), n=self.batch_size)

        pbar = tqdm(total=len(requests))
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            batched_inps = utils.pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long
                ).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))
                print(answer)

                res.append(answer)

                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    # def get_result(self, logprobs, context_length):
    #     is_greedy = True
    #     offsets = logprobs["text_offset"]
    #     tokens = logprobs["tokens"]
    #     tokens_logprobs = logprobs["token_logprobs"]

    #     idx = 0
    #     while offsets[idx] < context_length:
    #         idx += 1
    #     continuation_logprobs = sum(tokens_logprobs[idx:-1])
    #     for i in range(idx, len(tokens)):
    #         token = tokens[i]
    #         top_tokens = logprobs["top_logprobs"][i]
    #         top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
    #         if top_token != token:
    #             is_greedy = False
    #             break

    #     return continuation_logprobs, is_greedy

    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        pass

    # def loglikelihood(self, requests):
    #     if not requests:
    #         return []
    #     res = []
    #     for context, continuation in tqdm([req.args for req in requests]):
    #         response = self.gguf_completion(context=context, continuation=continuation)
    #         if response and "choices" in response and response["choices"]:
    #             choice = response["choices"][0]
    #             logprobs = choice.get("logprobs")
    #             if (
    #                 logprobs
    #                 and "token_logprobs" in logprobs
    #                 and logprobs["token_logprobs"]
    #             ):
    #                 logprob, is_greedy = get_result(logprobs, len(context))
    #                 res.append((logprob, is_greedy))
    #             else:
    #                 logger.warning(
    #                     "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
    #                 )
    #         else:
    #             logger.error(
    #                 f"Invalid response for loglikelihood. Response: {response}"
    #             )
    #             assert False
    #     return res

    def _model_call(self, inps) -> torch.Tensor:
        """
        Override the _model_call method to use the DeepSparse pipeline for
        logits generation.
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        # Encode the tokens to strings
        prompt = self.model.tokenizer.batch_decode(inps.numpy())
        print(prompt)

        # Run the model to map the prompt to logits
        out = self.model(
            prompt=prompt,
            max_new_tokens=0,
            include_prompt_logits=True,
            output_scores=True,
        )
        logits_numpy = numpy.stack([generation.score for generation in out.generations])
        return torch.from_numpy(logits_numpy)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests))
        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = utils.chunks(re_ord.get_reordered(), n=self.batch_size)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )
                
                if not until:
                    until = [self.tok_decode(self.eot_token_id)]

                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # we require users to pass do_sample=True explicitly for non-greedy gen
                if "do_sample" not in kwargs.keys():
                    kwargs["do_sample"] = False

                # first stop sequence is used to halt generation upon encountering
                primary_until = [until[0]]

                responses = self.model(
                    sequences=contexts,
                    max_new_tokens=max_gen_toks,
                    stop=until,
                    **kwargs,
                )

                responses = responses if type(responses) is list else [responses]
                for response, context in zip(responses, contexts):
                    text = response.generations[0].text
                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore possible empty separators
                            text = text.split(term)[0]
                    
                    res[key].append(text)
                    self.cache_hook.add_partial("greedy_until", (context, gen_kwargs), text)
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        pass

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits
