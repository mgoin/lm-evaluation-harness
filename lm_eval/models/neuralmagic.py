from pathlib import Path
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("sparseml")
class SparseMLLM(HFLM):
    """
    SparseML is an open-source model optimization toolkit that enables you to create
    inference-optimized sparse models using pruning, quantization, and distillation
    algorithms. Models optimized with SparseML can then be exported to the ONNX and
    deployed with DeepSparse for GPU-class performance on CPU hardware.
    """

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        try:
            import sparseml
        except ModuleNotFoundError:
            raise Exception(
                "package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        # Load model with SparseAutoModel
        from sparseml.transformers.utils import SparseAutoModel
        from transformers import AutoConfig

        model_kwargs = kwargs if kwargs else {}
        ignored_kwargs = [
            "dtype",
            "parallelize",
            "device_map_option",
            "max_memory_per_gpu",
            "max_cpu_memory",
            "peft",
            "autogptq",
        ]
        for k in ignored_kwargs:
            model_kwargs.pop(k)

        config = AutoConfig.from_pretrained(pretrained)
        model = SparseAutoModel.text_generation_from_pretrained(
            pretrained, config=config, **model_kwargs
        )

        # Apply recipe to model
        # Note: Really annoying we can't grab the recipe.yaml present in the uploaded model
        # and you need this separate apply_recipe_structure_to_model function
        from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
        from huggingface_hub import hf_hub_download
        import os

        recipe_path = hf_hub_download(repo_id=pretrained, filename="recipe.yaml")
        apply_recipe_structure_to_model(
            model=model,
            recipe_path=recipe_path,
            model_path=os.path.dirname(recipe_path),
        )

        self._model = model
