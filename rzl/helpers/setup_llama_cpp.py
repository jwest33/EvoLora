#!/usr/bin/env python
"""Setup script to download and configure llama.cpp for GGUF conversion on Windows"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import shutil
import subprocess
import zipfile
import urllib.request
from pathlib import Path
from utils.cli_formatter import CLIFormatter


def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress indicator

    Args:
        url: URL to download from
        destination: Path to save file
        description: Description for progress
    """
    CLIFormatter.print_info(f"{description}...")

    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f'\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print()
        CLIFormatter.print_error(f"Download failed: {e}")
        return False


def setup_llama_cpp_windows():
    """Setup llama.cpp on Windows by downloading pre-built binaries"""

    CLIFormatter.print_header("Setup llama.cpp for Windows")

    # Create tools directory
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)

    llama_dir = tools_dir / "llama.cpp"

    # Check if already exists
    if llama_dir.exists() and (llama_dir / "convert-hf-to-gguf.py").exists():
        CLIFormatter.print_info("llama.cpp already installed")

        response = input("Reinstall? (y/n): ").strip().lower()
        if response != 'y':
            CLIFormatter.print_info("Setup cancelled")
            return llama_dir

        shutil.rmtree(llama_dir)

    CLIFormatter.print_subheader("Step 1: Downloading llama.cpp")

    # Clone repository for Python scripts
    CLIFormatter.print_info("Cloning llama.cpp repository...")

    try:
        result = subprocess.run([
            "git", "clone",
            "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_dir)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")

        CLIFormatter.print_success("Repository cloned successfully")

    except FileNotFoundError:
        CLIFormatter.print_error("Git not found. Please install Git for Windows")
        CLIFormatter.print_info("Download from: https://git-scm.com/download/win")
        return None

    except Exception as e:
        CLIFormatter.print_error(f"Clone failed: {e}")
        return None

    CLIFormatter.print_subheader("Step 2: Downloading pre-built Windows binaries")

    # Get latest release info
    CLIFormatter.print_info("Fetching latest release...")

    release_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

    try:
        import json
        with urllib.request.urlopen(release_url) as response:
            release_data = json.loads(response.read())

        # Find Windows binary
        win_binary = None
        for asset in release_data['assets']:
            if 'win' in asset['name'].lower() and 'cuda' in asset['name'].lower():
                win_binary = asset
                break

        if not win_binary:
            # Fallback to non-CUDA version
            for asset in release_data['assets']:
                if 'win' in asset['name'].lower():
                    win_binary = asset
                    break

        if not win_binary:
            CLIFormatter.print_warning("No Windows binary found in latest release")
            CLIFormatter.print_info("You'll need to build from source")
            return llama_dir

        # Download binary
        binary_path = tools_dir / win_binary['name']

        if download_file(
            win_binary['browser_download_url'],
            binary_path,
            f"Downloading {win_binary['name']}"
        ):
            CLIFormatter.print_success("Binary downloaded successfully")

            # Extract if it's a zip
            if binary_path.suffix == '.zip':
                CLIFormatter.print_info("Extracting binaries...")

                with zipfile.ZipFile(binary_path, 'r') as zip_ref:
                    zip_ref.extractall(llama_dir)

                CLIFormatter.print_success("Extraction complete")

                # Clean up zip
                binary_path.unlink()

    except Exception as e:
        CLIFormatter.print_warning(f"Failed to download binaries: {e}")
        CLIFormatter.print_info("You can still use Python conversion scripts")

    CLIFormatter.print_subheader("Step 3: Installing Python dependencies")

    # Install required Python packages
    packages = ["gguf", "numpy", "sentencepiece", "protobuf", "torch", "transformers"]

    for package in packages:
        CLIFormatter.print_info(f"Checking {package}...")

        try:
            __import__(package.replace("-", "_"))
            CLIFormatter.print_success(f"{package} already installed")
        except ImportError:
            CLIFormatter.print_info(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])

    CLIFormatter.print_subheader("Setup Complete!")

    # Test installation
    convert_script = llama_dir / "convert-hf-to-gguf.py"
    quantize_exe = llama_dir / "llama-quantize.exe"

    if convert_script.exists():
        CLIFormatter.print_success(f"✓ Conversion script: {convert_script}")
    else:
        CLIFormatter.print_warning("Conversion script not found")

    if quantize_exe.exists():
        CLIFormatter.print_success(f"✓ Quantize tool: {quantize_exe}")
    else:
        CLIFormatter.print_warning("Quantize tool not found (build from source if needed)")

    # Update export script to use this installation
    CLIFormatter.print_info("\nUpdating export script paths...")

    export_script = Path("export_to_gguf.py")
    if export_script.exists():
        content = export_script.read_text()

        # Update llama.cpp paths
        if "llama_cpp_paths = [" in content:
            new_path = f'Path("{llama_dir}"),'
            if str(llama_dir) not in content:
                content = content.replace(
                    "llama_cpp_paths = [",
                    f"llama_cpp_paths = [\n        {new_path}"
                )
                export_script.write_text(content)
                CLIFormatter.print_success("Export script updated")

    return llama_dir


def setup_alternative_converter():
    """Setup alternative GGUF converter that doesn't require llama.cpp"""

    CLIFormatter.print_subheader("Alternative: Using Python-only converter")

    # Install llama-cpp-python with pip
    CLIFormatter.print_info("Installing llama-cpp-python...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python", "--upgrade",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"
        ], check=True)

        CLIFormatter.print_success("llama-cpp-python installed")

    except subprocess.CalledProcessError as e:
        CLIFormatter.print_warning("CUDA version failed, trying CPU version...")

        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "llama-cpp-python", "--upgrade"
            ], check=True)

            CLIFormatter.print_success("llama-cpp-python (CPU) installed")

        except subprocess.CalledProcessError:
            CLIFormatter.print_error("Failed to install llama-cpp-python")
            return False

    return True


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup llama.cpp for GGUF conversion on Windows"
    )

    parser.add_argument(
        "--alternative",
        action="store_true",
        help="Use alternative Python-only converter"
    )

    args = parser.parse_args()

    if args.alternative:
        if setup_alternative_converter():
            CLIFormatter.print_success("\nAlternative converter ready!")
            CLIFormatter.print_info("You can now run: python export_to_gguf.py")
    else:
        llama_dir = setup_llama_cpp_windows()
        if llama_dir:
            CLIFormatter.print_success(f"\nllama.cpp installed at: {llama_dir}")
            CLIFormatter.print_info("You can now run: python export_to_gguf.py")
        else:
            CLIFormatter.print_error("Setup failed")
            CLIFormatter.print_info("Try: python setup_llama_cpp.py --alternative")


if __name__ == "__main__":
    main()
