"""
PRIME-RL Synthetic Data Generation on Modal

This script runs synthetic data generation using a vLLM server on Modal.
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("prime-synthetic")

# Create shared volume for outputs
volume = modal.Volume.from_name("prime-rl-data", create_if_missing=True)
VOLUME_PATH = "/data"

# Define the Modal image with dependencies
def build_image():
    return (
        modal.Image.from_registry(
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
            add_python="3.12"
        )
        .run_commands("echo 'LC_ALL=en_US.UTF-8' >> /etc/environment")
        .apt_install("build-essential", "curl", "git", "git-lfs")
        # Install uv
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh > /uv-installer.sh",
            "INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR='/usr/local/bin' sh /uv-installer.sh",
            "rm /uv-installer.sh"
        )
        # Install prime CLI
        .run_commands("uv tool install prime")
        .workdir("/app")
        .add_local_dir(
            ".",
            remote_path="/app",
            copy=True,
            ignore=[".venv/", "__pycache__/", "*.pyc", ".modal/", ".git/"]
        )
        .env({
            "UV_COMPILE_BYTECODE": "1",
            "UV_LINK_MODE": "copy",
        })
        .run_commands(
            "uv sync --no-dev",
            gpu="any"
        )
        # Pre-install the environment
        .run_commands(
            "/root/.local/bin/prime env install primeintellect/acereason-math --with uv",
        )
        # Set runtime environment variables including PATH for prime CLI
        .env({
            "PATH": "/app/.venv/bin:/root/.local/bin:$PATH",
        })
    )

image = build_image()

# Configuration
GPU_CONFIG = "H100:1" # Adjust based on model size

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")] # Requires HF token for pushing
)
def generate_data(config_path: str = "configs/custom/synthesize_acereason.toml", push_to_hub: bool = False, hub_repo_id: str = "pmahdavi/acereason-synthetic-data"):
    """
    Starts a vLLM server in the background and runs the synthesis script.
    """
    import subprocess
    import time
    import sys
    import socket
    import tomli

    # Read model name from config file
    config_file = Path("/app") / config_path
    with open(config_file, "rb") as f:
        config = tomli.load(f)
    model_name = config.get("model", {}).get("name", "Qwen/Qwen2.5-Math-7B-Instruct")
    print(f"Model name from config: {model_name}")

    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = 8000 # Fixed port for simplicity in config, or use get_free_port() and update config dynamically

    print(f"Starting vLLM server for {model_name} on port {port}...")
    
    # 1. Start vLLM Server in background
    vllm_cmd = [
        "uv", "run", "vllm", "serve", model_name,
        "--port", str(port),
        "--trust-remote-code",
        # Remove --max-model-len to use model's default (4096 for Qwen2.5-Math)
        # Or set to model's max: "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto"
    ]
    
    # Add --enable-reasoning if using a reasoning model like DeepSeek-R1
    # vllm_cmd.append("--enable-reasoning")

    server_process = subprocess.Popen(
        vllm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        bufsize=1 # Line buffered
    )

    # 2. Wait for server to be ready
    print("Waiting for vLLM server to be ready...")
    # Simple health check loop
    import requests
    for i in range(60): # Wait up to 10 minutes
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print("vLLM server is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(10)
        if server_process.poll() is not None:
            print("vLLM server failed to start.")
            return

    # 3. Run Synthesis
    print(f"Running synthesis with config: {config_path}")
    
    # We need to make sure the config points to localhost:{port}
    # Since we fixed port 8000, the existing config should work if it uses localhost:8000
    
    synth_cmd = [
        "uv", "run", "synthesize",
        "@", config_path,
        "--output-dir", "/data/outputs/synthetic_data" # Save to volume
    ]
    
    try:
        subprocess.run(synth_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("Synthesis complete.")
    except subprocess.CalledProcessError as e:
        print(f"Synthesis failed with code {e.returncode}")
    finally:
        # Cleanup
        print("Stopping vLLM server...")
        server_process.terminate()
        server_process.wait()

    # 4. Push to Hub (Optional)
    if push_to_hub and hub_repo_id:
        print(f"Pushing dataset to HF Hub: {hub_repo_id}")
        
        # Find the results file. Structure: /data/outputs/synthetic_data/{env_id}/{model_name}/results.jsonl
        # We search recursively
        results_files = list(Path("/data/outputs/synthetic_data").rglob("results.jsonl"))
        if not results_files:
            print("No results.jsonl found to push.")
            return

        print(f"Found {len(results_files)} result files. merging and pushing...")
        
        # Use uv run python to ensure we're in the right environment with datasets installed
        import json
        results_files_str = json.dumps([str(f) for f in results_files])
        push_script = f"""from datasets import load_dataset
import json as json_lib
results_files = {results_files_str}
ds = load_dataset("json", data_files=results_files, split="train")

# Fix empty struct fields that PyArrow can't handle
# Convert 'info' dict field to JSON string to avoid Parquet struct issues
def fix_info_field(example):
    if 'info' in example and isinstance(example['info'], dict):
        example['info'] = json_lib.dumps(example['info'])
    return example

ds = ds.map(fix_info_field)
ds.push_to_hub("{hub_repo_id}")
print("Successfully pushed to https://huggingface.co/datasets/{hub_repo_id}")
"""
        push_cmd = ["uv", "run", "python", "-c", push_script]
        
        try:
            env = os.environ.copy()
            env["PATH"] = f"/app/.venv/bin:/root/.local/bin:{env.get('PATH', '')}"
            subprocess.run(push_cmd, check=True, cwd="/app", env=env)
        except subprocess.CalledProcessError as e:
            print(f"Failed to push to hub: {e}")


