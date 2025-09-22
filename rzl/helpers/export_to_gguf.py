#!/usr/bin/env python
"""Export trained R-Zero model to GGUF format for llama.cpp inference"""

import os
import sys
import torch
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from unsloth import FastModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cli_formatter import CLIFormatter, Style

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def find_latest_checkpoint(base_dir: str = "outputs/solver") -> Optional[str]:
    """Find the latest training checkpoint

    Args:
        base_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    solver_dir = Path(base_dir)
    if not solver_dir.exists():
        return None

    iterations = [d for d in solver_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
    if not iterations:
        return None

    latest = max(iterations, key=lambda x: int(x.name.split("_")[1]))
    return str(latest)


def export_model_to_gguf(
    checkpoint_path: Optional[str] = None,
    base_model: str = "unsloth/gemma-3-1b-it-bnb-4bit",
    output_dir: str = "outputs/gguf",
    quantization: str = "q8_0",
    model_name: str = "rzero_solver"
) -> Optional[str]:
    """Export a trained model to GGUF format

    Args:
        checkpoint_path: Path to checkpoint (None = use latest)
        base_model: Base model name if no checkpoint
        output_dir: Output directory for GGUF files
        quantization: Quantization method (q8_0, q6_k, q5_k_m, q4_k_m, q4_0, f16)
        model_name: Name prefix for output file

    Returns:
        Path to exported GGUF file or None if failed
    """
    CLIFormatter.print_header("Export Model to GGUF Format")

    # Find checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            CLIFormatter.print_info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            CLIFormatter.print_warning("No checkpoints found, using base model")

    # Load the model
    CLIFormatter.print_subheader("Step 1: Loading Model")

    try:
        if checkpoint_path and Path(checkpoint_path).exists():
            CLIFormatter.print_info(f"Loading checkpoint: {checkpoint_path}")
            model, tokenizer = FastModel.from_pretrained(
                model_name=checkpoint_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                load_in_4bit=True,
            )
        else:
            CLIFormatter.print_info(f"Loading base model: {base_model}")
            model, tokenizer = FastModel.from_pretrained(
                model_name=base_model,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                load_in_4bit=True,
            )
    except Exception as e:
        CLIFormatter.print_error(f"Failed to load model: {e}")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save in HuggingFace format first
    CLIFormatter.print_subheader("Step 2: Preparing for Conversion")

    temp_dir = Path(output_dir) / "temp_export"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        CLIFormatter.print_info(f"Saving to temporary directory: {temp_dir}")

        # Save merged model (combines LoRA weights with base model)
        model.save_pretrained_merged(
            str(temp_dir),
            tokenizer,
            save_method="merged_16bit"
        )

        CLIFormatter.print_success("Model prepared for conversion")

    except Exception as e:
        CLIFormatter.print_error(f"Failed to save model: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None

    # Convert to GGUF
    CLIFormatter.print_subheader("Step 3: Converting to GGUF")

    output_file = Path(output_dir) / f"{model_name}_{quantization}.gguf"

    # Try direct conversion using llama-cpp-python's convert tools
    try:
        CLIFormatter.print_info("Converting to GGUF format...")

        # First try using llama-cpp-python's built-in converter if available
        try:
            from gguf import GGUFWriter
            from transformers.models.llama import convert_llama_weights_to_hf
            CLIFormatter.print_info("Using gguf package for conversion...")

            # This is a simplified path - for full conversion we need llama.cpp tools
            raise ImportError("Need full llama.cpp for proper conversion")

        except ImportError:
            pass

        # Try Unsloth's built-in GGUF export
        CLIFormatter.print_info("Attempting Unsloth GGUF export...")

        # Map quantization names for Unsloth
        quant_map = {
            "q8_0": "Q8_0",
            "q6_k": "Q6_K",
            "q5_k_m": "Q5_K_M",
            "q4_k_m": "Q4_K_M",
            "q4_0": "Q4_0",
            "f16": "F16"
        }

        # Suppress pip warnings by setting environment variable

        os.environ["PIP_NO_WARN_SCRIPT_LOCATION"] = "0"
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        model.save_pretrained_gguf(
            str(temp_dir),
            tokenizer,
            quantization_method=quant_map.get(quantization, "Q8_0")
        )

        # Find and move the generated GGUF file
        gguf_files = list(temp_dir.glob("*.gguf"))
        if gguf_files:
            shutil.move(str(gguf_files[0]), str(output_file))
            CLIFormatter.print_success(f"GGUF export successful!")
        else:
            raise Exception("No GGUF file was generated")

    except Exception as e:
        CLIFormatter.print_warning(f"Unsloth export method failed: {e}")
        CLIFormatter.print_info("Trying alternative conversion method...")

        # Manual conversion using llama.cpp
        if not convert_with_llama_cpp(temp_dir, output_file, quantization):
            CLIFormatter.print_error("Manual conversion also failed")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None

    # Clean up temporary files
    CLIFormatter.print_info("Cleaning up...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Report success and file info
    if output_file.exists():
        file_size_gb = output_file.stat().st_size / (1024 ** 3)

        CLIFormatter.print_subheader("Export Complete!")
        CLIFormatter.print_success(f"✓ Model exported to: {output_file}")
        CLIFormatter.print_status("Quantization", quantization)
        CLIFormatter.print_status("File size", f"{file_size_gb:.2f} GB")

        return str(output_file)
    else:
        CLIFormatter.print_error("Export failed - no output file created")
        return None


def convert_with_llama_cpp(
    model_dir: Path,
    output_file: Path,
    quantization: str
) -> bool:
    """Convert model using llama.cpp tools

    Args:
        model_dir: Directory containing HF model
        output_file: Output GGUF file path
        quantization: Quantization method

    Returns:
        True if successful, False otherwise
    """
    # Common paths for llama.cpp installation
    llama_cpp_paths = [
        Path("tools\llama.cpp"),
        Path("tools/llama.cpp"),  # Our setup location
        Path("C:/llama.cpp"),
        Path("~/llama.cpp").expanduser(),
        Path("./llama.cpp"),
        Path("../llama.cpp"),
    ]

    # Find llama.cpp installation
    llama_cpp_dir = None
    for path in llama_cpp_paths:
        # Check for new or old convert script names (with hyphens or underscores)
        if path.exists() and (
            (path / "convert_hf_to_gguf.py").exists() or
            (path / "convert-hf-to-gguf.py").exists() or
            (path / "convert.py").exists()
        ):
            llama_cpp_dir = path
            break

    if not llama_cpp_dir:
        CLIFormatter.print_error("llama.cpp not found!")
        CLIFormatter.print_info("Please run setup first:")
        CLIFormatter.print_info("  python setup_llama_cpp.py")
        return False

    # Find the convert script (try all possible names)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"  # Most common
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert.py"

    # Find quantize executable (try different names)
    quantize_exe = llama_cpp_dir / "llama-quantize.exe"
    if not quantize_exe.exists():
        quantize_exe = llama_cpp_dir / "quantize.exe"

    # Check if convert script exists
    if not convert_script.exists():
        CLIFormatter.print_error(f"Convert script not found: {convert_script}")
        return False

    CLIFormatter.print_info(f"Using convert script: {convert_script.name}")

    # Convert to GGUF directly (new scripts don't need F16 intermediate)
    CLIFormatter.print_info(f"Converting to GGUF {quantization.upper()} format...")

    # Build the command based on the script type
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile", str(output_file),
        "--outtype", quantization.lower()
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        CLIFormatter.print_error(f"Conversion error: {result.stderr}")

        # If the error is about outtype, try without it (older script format)
        if "--outtype" in result.stderr or "unrecognized arguments" in result.stderr:
            CLIFormatter.print_info("Retrying with older script format...")

            # Older scripts might not support --outtype, convert to F16 first
            f16_file = output_file.parent / f"{output_file.stem}_f16.gguf"
            cmd = [
                sys.executable,
                str(convert_script),
                str(model_dir),
                "--outfile", str(f16_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                CLIFormatter.print_error(f"Conversion still failed: {result.stderr}")
                return False

            # Now we need to quantize if not F16
            if quantization != "f16":
                if quantize_exe.exists():
                    CLIFormatter.print_info(f"Quantizing to {quantization}...")
                    result = subprocess.run([
                        str(quantize_exe),
                        str(f16_file),
                        str(output_file),
                        quantization
                    ], capture_output=True, text=True)

                    if result.returncode == 0:
                        f16_file.unlink(missing_ok=True)
                    else:
                        CLIFormatter.print_error(f"Quantization error: {result.stderr}")
                        f16_file.unlink(missing_ok=True)
                        return False
                else:
                    CLIFormatter.print_warning(f"Quantize tool not found, output will be F16")
                    f16_file.rename(output_file)
            else:
                f16_file.rename(output_file)
        else:
            return False

    return True


def main():
    """Main entry point for export script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export R-Zero trained model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization options:
  q8_0   - 8-bit quantization (best quality, ~1GB for 1B model)
  q6_k   - 6-bit quantization (very good, ~750MB)
  q5_k_m - 5-bit quantization (good, ~650MB)
  q4_k_m - 4-bit quantization (acceptable, ~550MB)
  q4_0   - 4-bit quantization (smaller, ~500MB)
  f16    - No quantization (largest, ~2GB)

Examples:
  python export_to_gguf.py                    # Export latest checkpoint with Q8_0
  python export_to_gguf.py --quantization q6_k  # Export with Q6_K quantization
  python export_to_gguf.py --checkpoint outputs/solver/iteration_3  # Export specific checkpoint
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint directory (default: use latest)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-1b-it-bnb-4bit",
        help="Base model name if no checkpoint found"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gguf",
        help="Output directory for GGUF files (default: outputs/gguf)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q8_0",
        choices=["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "f16"],
        help="Quantization method (default: q8_0)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rzero_solver",
        help="Output model name prefix (default: rzero_solver)"
    )

    args = parser.parse_args()

    # Run export
    output_file = export_model_to_gguf(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        output_dir=args.output_dir,
        quantization=args.quantization,
        model_name=args.name
    )

    if output_file:
        print()
        CLIFormatter.print_info("To chat with the model, run:")
        CLIFormatter.print_info(f"  python chat_cli.py --model {output_file}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
