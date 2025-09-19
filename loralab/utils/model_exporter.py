"""Model export utilities for various formats

Supports exporting LoRA models to different formats including:
- GGUF for llama.cpp
- Merged 16-bit and 4-bit models
- Ollama format
- HuggingFace Hub
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model export to various formats"""

    def __init__(self, model_manager):
        """Initialize exporter with model manager

        Args:
            model_manager: UnslothModelManager or ModelManager instance
        """
        self.model_manager = model_manager
        self.tokenizer = model_manager.get_tokenizer()

        # Check if using Unsloth
        self.is_unsloth = hasattr(model_manager, 'save_model')

    def export_lora(self, model: Any, save_path: str,
                   push_to_hub: bool = False, hub_id: str = None,
                   token: str = None) -> str:
        """Export LoRA adapters only

        Args:
            model: Model with LoRA adapters
            save_path: Local save path
            push_to_hub: Whether to push to HuggingFace Hub
            hub_id: HuggingFace Hub repository ID
            token: HuggingFace token

        Returns:
            Path where model was saved
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.is_unsloth:
            # Use Unsloth's save method
            self.model_manager.save_model(model, str(save_path), "lora")
        else:
            # Standard PEFT save
            model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Saved LoRA adapters to {save_path}")

        if push_to_hub and hub_id:
            try:
                model.push_to_hub(hub_id, token=token)
                self.tokenizer.push_to_hub(hub_id, token=token)
                logger.info(f"Pushed LoRA to Hub: {hub_id}")
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")

        return str(save_path)

    def export_merged(self, model: Any, save_path: str,
                     quantization: str = "16bit",
                     push_to_hub: bool = False, hub_id: str = None,
                     token: str = None) -> str:
        """Export merged model (base + LoRA)

        Args:
            model: Model with LoRA adapters
            save_path: Local save path
            quantization: "16bit" or "4bit"
            push_to_hub: Whether to push to HuggingFace Hub
            hub_id: HuggingFace Hub repository ID
            token: HuggingFace token

        Returns:
            Path where model was saved
        """
        save_path = Path(save_path)

        if self.is_unsloth:
            # Use Unsloth's optimized merge and save
            save_method = f"merged_{quantization.replace('bit', '')}bit"
            self.model_manager.save_model(model, str(save_path), save_method)
            logger.info(f"Saved merged {quantization} model to {save_path}")

            if push_to_hub and hub_id:
                try:
                    # Unsloth's push_to_hub_merged method
                    model.push_to_hub_merged(
                        hub_id,
                        self.tokenizer,
                        save_method=save_method,
                        token=token
                    )
                    logger.info(f"Pushed merged model to Hub: {hub_id}")
                except Exception as e:
                    logger.error(f"Failed to push to Hub: {e}")
        else:
            # Standard merge and save
            logger.warning("Merged export requires Unsloth. Saving LoRA only.")
            return self.export_lora(model, save_path, push_to_hub, hub_id, token)

        return str(save_path)

    def export_gguf(self, model: Any, save_path: str,
                   quantization: str = "q8_0",
                   push_to_hub: bool = False, hub_id: str = None,
                   token: str = None) -> str:
        """Export model in GGUF format for llama.cpp

        Args:
            model: Model to export
            save_path: Local save path
            quantization: GGUF quantization method (q8_0, q4_k_m, q5_k_m, etc.)
            push_to_hub: Whether to push to HuggingFace Hub
            hub_id: HuggingFace Hub repository ID
            token: HuggingFace token

        Returns:
            Path where model was saved
        """
        save_path = Path(save_path)

        if not self.is_unsloth:
            logger.error("GGUF export requires Unsloth")
            raise ValueError("GGUF export requires Unsloth. Install with: pip install unsloth")

        # Use Unsloth's GGUF export
        self.model_manager.save_model(model, str(save_path), f"gguf_{quantization}")
        logger.info(f"Saved GGUF model ({quantization}) to {save_path}")

        if push_to_hub and hub_id:
            try:
                model.push_to_hub_gguf(
                    hub_id,
                    self.tokenizer,
                    quantization_method=quantization,
                    token=token
                )
                logger.info(f"Pushed GGUF model to Hub: {hub_id}")
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")

        return str(save_path)

    def export_multiple_gguf(self, model: Any, save_path: str,
                           quantizations: List[str] = None,
                           push_to_hub: bool = False, hub_id: str = None,
                           token: str = None) -> Dict[str, str]:
        """Export model in multiple GGUF quantization formats

        Args:
            model: Model to export
            save_path: Base save path
            quantizations: List of quantization methods
            push_to_hub: Whether to push to HuggingFace Hub
            hub_id: HuggingFace Hub repository ID
            token: HuggingFace token

        Returns:
            Dictionary mapping quantization to save path
        """
        if quantizations is None:
            quantizations = ["q8_0", "q4_k_m", "q5_k_m"]

        save_paths = {}

        if push_to_hub and hub_id and self.is_unsloth:
            # Batch export to Hub (more efficient)
            try:
                model.push_to_hub_gguf(
                    hub_id,
                    self.tokenizer,
                    quantization_method=quantizations,
                    token=token
                )
                logger.info(f"Pushed multiple GGUF formats to Hub: {hub_id}")
                for quant in quantizations:
                    save_paths[quant] = f"{hub_id}/model-unsloth-{quant.upper()}.gguf"
            except Exception as e:
                logger.error(f"Failed to push to Hub: {e}")
        else:
            # Export individually
            for quant in quantizations:
                quant_path = Path(save_path) / quant
                save_paths[quant] = self.export_gguf(
                    model, str(quant_path), quant, False, None, None
                )

        return save_paths

    def export_for_ollama(self, model: Any, model_name: str,
                        quantization: str = "q4_k_m",
                        create_modelfile: bool = True) -> str:
        """Export model for Ollama

        Args:
            model: Model to export
            model_name: Name for Ollama model
            quantization: GGUF quantization method
            create_modelfile: Whether to create Ollama Modelfile

        Returns:
            Path to exported model
        """
        ollama_dir = Path("models/ollama") / model_name
        ollama_dir.mkdir(parents=True, exist_ok=True)

        # Export as GGUF
        gguf_path = self.export_gguf(model, str(ollama_dir), quantization)

        if create_modelfile:
            # Create Ollama Modelfile
            modelfile_content = f"""# Modelfile for {model_name}
# Generated by LoRALab

FROM ./model-unsloth.gguf

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt (customize as needed)
SYSTEM "You are a helpful AI assistant."

# Template (optional, customize based on model)
# TEMPLATE "..."
"""
            modelfile_path = ollama_dir / "Modelfile"
            modelfile_path.write_text(modelfile_content)
            logger.info(f"Created Modelfile at {modelfile_path}")

            logger.info(f"\nTo use with Ollama:")
            logger.info(f"  cd {ollama_dir}")
            logger.info(f"  ollama create {model_name} -f Modelfile")
            logger.info(f"  ollama run {model_name}")

        return str(ollama_dir)

    def export_best_variant(self, model: Any, variant_info: Dict[str, Any],
                          output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """Export the best variant in multiple formats

        Args:
            model: Best model variant
            variant_info: Information about the variant (rank, alpha, etc.)
            output_dir: Base output directory
            formats: List of formats to export ("lora", "merged_16bit", "gguf_q4_k_m", etc.)

        Returns:
            Dictionary mapping format to save path
        """
        if formats is None:
            formats = ["lora", "merged_16bit"]

        output_dir = Path(output_dir)
        variant_name = f"best_r{variant_info.get('rank', 'unknown')}_gen{variant_info.get('generation', 'unknown')}"

        export_paths = {}

        for format_type in formats:
            format_dir = output_dir / format_type / variant_name

            if format_type == "lora":
                export_paths[format_type] = self.export_lora(model, str(format_dir))

            elif format_type.startswith("merged_"):
                quantization = format_type.split("_")[1]
                export_paths[format_type] = self.export_merged(
                    model, str(format_dir), quantization
                )

            elif format_type.startswith("gguf_"):
                quantization = format_type.split("_", 1)[1]
                export_paths[format_type] = self.export_gguf(
                    model, str(format_dir), quantization
                )

            elif format_type == "ollama":
                export_paths[format_type] = self.export_for_ollama(
                    model, variant_name, "q4_k_m"
                )

            else:
                logger.warning(f"Unknown export format: {format_type}")

        # Save variant info
        info_path = output_dir / f"{variant_name}_info.json"
        import json
        with open(info_path, 'w') as f:
            json.dump(variant_info, f, indent=2)
        logger.info(f"Saved variant info to {info_path}")

        return export_paths

    def cleanup_old_exports(self, output_dir: str, keep_best: int = 3):
        """Clean up old exported models to save space

        Args:
            output_dir: Directory containing exports
            keep_best: Number of best models to keep
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return

        # Find all variant directories
        variant_dirs = []
        for format_dir in output_dir.iterdir():
            if format_dir.is_dir():
                for variant_dir in format_dir.iterdir():
                    if variant_dir.is_dir():
                        # Try to extract generation number from name
                        try:
                            gen_num = int(variant_dir.name.split("_gen")[1].split("_")[0])
                            variant_dirs.append((gen_num, variant_dir))
                        except:
                            pass

        # Sort by generation number (newest first)
        variant_dirs.sort(reverse=True)

        # Remove old variants
        for i, (gen_num, variant_dir) in enumerate(variant_dirs):
            if i >= keep_best:
                try:
                    shutil.rmtree(variant_dir)
                    logger.info(f"Removed old export: {variant_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove {variant_dir}: {e}")


# Utility function for CLI
def export_cli_model(model_path: str, export_format: str, output_path: str,
                   quantization: Optional[str] = None, **kwargs):
    """CLI utility for model export

    Args:
        model_path: Path to model to export
        export_format: Export format (lora, merged, gguf, ollama)
        output_path: Output path
        quantization: Quantization level (for merged/gguf)
        **kwargs: Additional arguments
    """
    # This would be called from a CLI script
    logger.info(f"Exporting {model_path} as {export_format} to {output_path}")
    # Implementation would load model and call appropriate export method