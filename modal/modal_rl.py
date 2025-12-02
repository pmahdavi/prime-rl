"""
PRIME-RL on Modal

This script enables running PRIME-RL training on Modal's serverless GPU infrastructure.

Usage:
    # Run any command (RL training, eval, inference, etc.)
    modal run modal_rl.py::run_command --cmd "uv run rl @ configs/gsm8k/rl.toml --wandb.project gsm8k-modal --wandb.name gsm8k-run --ckpt"
    
    # Read logs during training
    modal run modal_rl.py::run_command --cmd "tail -n 100 /data/outputs/logs/orchestrator.stdout"
    
    # Stream logs in real-time
    modal run modal_rl.py::run_command --cmd "tail -f /data/outputs/logs/trainer.stdout"

    # Run tests
    modal run modal_rl.py::run_tests

    # Run pre-commit hooks
    modal run modal_rl.py::run_pre_commit
"""

import modal
import os

# Create Modal app
app = modal.App("prime-rl")

# Create shared volume for checkpoints, weights, and outputs
volume = modal.Volume.from_name("prime-rl-data", create_if_missing=True)
VOLUME_PATH = "/data"

# Define the Modal image with all dependencies
def build_image():
    """
    Build the Modal container image with all PRIME-RL dependencies.
    """
    return (
        # Use PyTorch CUDA base image
        modal.Image.from_registry(
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
            add_python="3.12"
        )
        # Set locale
        .run_commands("echo 'LC_ALL=en_US.UTF-8' >> /etc/environment")
        # Install system dependencies
        .apt_install(
            "build-essential",
            "curl",
            "sudo",
            "git",
            "git-lfs",
        )
        # Install uv
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh > /uv-installer.sh",
            "INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR='/usr/local/bin' sh /uv-installer.sh",
            "rm /uv-installer.sh"
        )
        # Install prime CLI for environment management
        .run_commands(
            "uv tool install prime",
        )
        # Set working directory and copy repository
        .workdir("/app")
        .add_local_dir(
            ".",
            remote_path="/app",
            copy=True,
            ignore=[".venv/", "__pycache__/", "*.pyc", ".modal/"]
        )
        # Install dependencies
        .env({
            "UV_COMPILE_BYTECODE": "1",
            "UV_LINK_MODE": "copy",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "$PATH:/usr/local/cuda/bin",
        })
        .run_commands(
            "uv sync --no-dev",
            gpu="any"  # Some packages (flash-attn, vllm) require GPU during build
        )
        # Install environments from Prime Intellect hub
        .run_commands(
            "/root/.local/bin/prime env install primeintellect/single-turn-math --with uv",
        )
        # Set runtime environment variables
        .env({
            "PYTHONUNBUFFERED": "1",
            "PATH": "/app/.venv/bin:/root/.local/bin:$PATH",
        })
    )

image = build_image()

# Wandb secret for logging
wandb_secret = modal.Secret.from_name("wandb")

# GPU configuration (configurable via environment variables)
RL_TRAINING_GPU = os.getenv("RL_TRAINING_GPU", "H100:2")  # 2 GPUs for RL (inference + trainer)

# Timeout configuration (in seconds)
TIMEOUT = 7200  # 2 hours default timeout


@app.function(
    image=image,
    gpu=RL_TRAINING_GPU,  # 2 GPUs by default (can handle RL training, inference, or single-GPU commands)
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT * 2,  # Longer timeout to handle training runs
    secrets=[wandb_secret],
)
def run_command(cmd: str, output_dir: str | None = None, commit_volume: bool = True):
    """
    Run any command in the Modal environment.
    
    This is a unified function that handles all commands - RL training, eval, inference, etc.
    It automatically detects RL commands and adds --output-dir if needed.
    GPU usage is controlled by the command itself (e.g., CUDA_VISIBLE_DEVICES).
    
    Examples:
        # RL training
        modal run modal_rl.py::run_command --cmd "uv run rl @ configs/gsm8k/rl.toml --wandb.project gsm8k-modal --wandb.name gsm8k-run --ckpt"
        
        # Evaluation (connects to inference server)
        modal run modal_rl.py::run_command --cmd "uv run vf-eval single-turn-math -a '{\"dataset_name\": \"openai/gsm8k\"}' -m PrimeIntellect/Qwen3-0.6B -b http://localhost:8000/v1 -n 20"
        
        # Inference server (use CUDA_VISIBLE_DEVICES to use only GPU 0)
        modal run modal_rl.py::run_command --cmd "CUDA_VISIBLE_DEVICES=0 uv run inference --model.name PrimeIntellect/Qwen3-0.6B"
        
        # Any Python command
        modal run modal_rl.py::run_command --cmd "uv run python -c 'import torch; print(torch.cuda.device_count())'"
        
        # List outputs
        modal run modal_rl.py::run_command --cmd "ls -lah /data/outputs"
        
        # Read logs
        modal run modal_rl.py::run_command --cmd "tail -n 100 /data/outputs/logs/orchestrator.stdout"
        
        # Stream logs
        modal run modal_rl.py::run_command --cmd "tail -f /data/outputs/logs/trainer.stdout"
    
    Args:
        cmd: The command to run
        output_dir: Output directory for RL commands (default: /data/outputs). Only used if cmd is an RL command.
        commit_volume: Whether to commit volume periodically during execution (default: True). Set False for quick commands.
    """
    import subprocess
    import sys
    import os
    import threading
    import time
    
    # Detect if this is an RL command
    is_rl_command = "uv run rl" in cmd or "rl @" in cmd
    
    # Append output-dir for RL commands if not present
    if is_rl_command and output_dir is None:
        output_dir = "/data/outputs"
    
    if is_rl_command and output_dir and "--output-dir" not in cmd:
        cmd = f"{cmd} --output-dir {output_dir}"
    
    print(f"Running: {cmd}")
    if is_rl_command:
        print(f"Output dir: {output_dir}")
    print(f"Working directory: /app")
    print(f"Python version: {sys.version}")
    
    # Verify uv is available
    uv_check = subprocess.run(["which", "uv"], capture_output=True, text=True)
    if uv_check.returncode == 0:
        print(f"uv found at: {uv_check.stdout.strip()}")
    else:
        print("WARNING: uv not found in PATH")
    
    if is_rl_command:
        print("Note: To access logs during training, use:")
        print("  modal run modal_rl.py::run_command --cmd 'tail -f /data/outputs/logs/trainer.stdout'")
        print("Available log files: trainer.stdout, orchestrator.stdout, inference.stdout, rl.log")
    
    print()
    sys.stdout.flush()
    
    # Set up environment to ensure unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Periodic volume commits for long-running commands
    commit_thread = None
    should_commit = threading.Event()
    
    if commit_volume:
        commit_interval = 2400  # 40 minutes
        
        def periodic_commit():
            """Commit volume periodically so data is accessible from other functions."""
            while not should_commit.is_set():
                time.sleep(commit_interval)
                if not should_commit.is_set():
                    try:
                        volume.commit()
                        print(f"\n[Volume committed at {time.strftime('%H:%M:%S')}] Data is now accessible from other functions.", flush=True)
                    except Exception as e:
                        print(f"\n[Warning] Failed to commit volume: {e}", flush=True)
        
        commit_thread = threading.Thread(target=periodic_commit, daemon=True)
        commit_thread.start()
    
    result = None
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd="/app",
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, terminating...")
        raise
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always commit volume if enabled, even on failure or interrupt
        if commit_volume:
            should_commit.set()
            if commit_thread:
                commit_thread.join(timeout=1)
            try:
                volume.commit()
                print()
                if result is not None:
                    print(f"Command completed with code: {result.returncode}")
                else:
                    print("Command was interrupted or failed, but volume was committed.")
                if is_rl_command:
                    print(f"Download outputs: modal volume get prime-rl-data outputs/ ./outputs/")
            except Exception as e:
                print(f"\n[Warning] Failed to commit volume: {e}", flush=True)
    
    if result is not None:
        return result.returncode
    else:
        return 1  # Return error code if command didn't complete


@app.function(
    image=image,
    gpu="H100:2",  # Use 2 H100s to support integration tests expecting multiple GPUs
    timeout=3600,  # 1 hour timeout
)
def run_tests():
    """
    Run the full test suite as described in the README.
    """
    import os
    # Run W&B in offline mode for tests
    os.environ["WANDB_MODE"] = "offline"
    
    print("Running full test suite...")
    
    # Run pytest
    cmd = "uv run pytest -v"
    
    import subprocess
    import sys
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed.")
        
    return result.returncode


@app.function(
    image=image,
    gpu="H100:2",  # Need 2 GPUs: one for vLLM server (GPU 0), one for trainer if needed
    timeout=3600,  # 1 hour timeout (integration tests can take time)
    volumes={VOLUME_PATH: volume},
    secrets=[wandb_secret],
)
def run_integration_tests():
    """
    Run all integration tests (matches CI exactly).
    
    This runs the exact same tests as CI:
    - All integration tests with GPU marker (except LoRA tests)
    - LoRA integration tests separately
    
    Tests include:
    - test_orchestrator: Orchestrator integration test
    - test_rl: Full RL training integration test
    - test_sft: SFT training integration test
    - test_trainer: Trainer integration test
    - test_eval: Evaluation integration test
    - test_client: Client integration test
    - test_rl (LoRA): LoRA RL training integration test
    
    Usage:
        modal run modal_rl.py::run_integration_tests
    """
    import subprocess
    import sys
    import os
    
    # Set environment variables (same as CI)
    os.environ["WANDB_MODE"] = "offline"  # Run W&B in offline mode for tests
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["USERNAME_CI"] = "CI_RUNNER"  # Match CI username
    
    # Set output directory for test (pytest fixture uses PYTEST_OUTPUT_DIR)
    output_dir = "/data/outputs"
    os.environ["PYTEST_OUTPUT_DIR"] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Running all integration tests (matching CI)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("Note: pytest fixtures will start/stop the vLLM server automatically")
    print()
    
    # Run integration tests (same as CI)
    # First: all integration tests except LoRA
    print("Running integration tests (excluding LoRA)...")
    print("-" * 80)
    cmd1 = "uv run pytest tests/integration -m gpu --ignore=tests/integration/lora/test_rl.py -v"
    result1 = subprocess.run(
        cmd1,
        shell=True,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=os.environ,
    )
    
    if result1.returncode != 0:
        print("\n" + "=" * 80)
        print("❌ Integration tests failed.")
        print(f"\nCheck logs at: {output_dir}/")
        print("=" * 80)
        return result1.returncode
    
    # Second: LoRA integration tests
    print("\n" + "-" * 80)
    print("Running LoRA integration tests...")
    print("-" * 80)
    cmd2 = "uv run pytest tests/integration/lora -m gpu -v"
    result2 = subprocess.run(
        cmd2,
        shell=True,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=os.environ,
    )
    
    print("\n" + "=" * 80)
    if result2.returncode == 0:
        print("✅ All integration tests passed!")
    else:
        print("❌ LoRA integration tests failed.")
        print(f"\nCheck logs at: {output_dir}/")
    print("=" * 80)
    
    return result2.returncode


@app.function(
    image=image,
    gpu="any",  # Use any available GPU
    timeout=1800,  # 30 minutes timeout
)
def run_pre_commit():
    """
    Run pre-commit hooks to ensure code quality.
    """
    print("Running pre-commit hooks...")
    
    # Install pre-commit hooks first (as per README)
    install_cmd = "uv run pre-commit install"
    run_cmd = "uv run pre-commit run --all-files"
    
    import subprocess
    import sys
    
    # Install hooks
    print(f"Installing hooks: {install_cmd}")
    subprocess.run(
        install_cmd,
        shell=True,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    
    # Run hooks
    print(f"\nRunning hooks: {run_cmd}")
    result = subprocess.run(
        run_cmd,
        shell=True,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ Pre-commit hooks passed!")
    else:
        print("\n❌ Pre-commit hooks failed.")
        
    return result.returncode
