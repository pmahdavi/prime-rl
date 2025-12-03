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

    Layer ordering is optimized for caching:
    1. Base image (rarely changes)
    2. System packages via apt_install (rarely changes)
    3. uv installation (rarely changes)
    4. prime CLI installation (rarely changes)
    5. Python dependencies via uv sync (only rebuilds when pyproject.toml/uv.lock change)
    6. Prime environments installation (rarely changes)
    7. Environment variables (must be before runtime mounts)
    8-12. Configs, source, tests, examples (mounted at RUNTIME - instant deploys!)

    Key optimizations:
    - Copy only pyproject.toml, uv.lock, README.md before uv sync
    - Use --no-install-project: source code isn't available at build time (mounted at runtime)
    - All project files use copy=False: mounted at container startup, not built into image
    - Result: code/config changes deploy instantly, no image rebuild needed

    IMPORTANT: Runtime mounts (copy=False) must be LAST - no build steps can follow them.
    """
    return (
        # Layer 1: Base image (rarely changes)
        modal.Image.from_registry(
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
            add_python="3.12"
        )
        # Layer 2: System dependencies (rarely changes)
        .run_commands("echo 'LC_ALL=en_US.UTF-8' >> /etc/environment")
        .apt_install(
            "build-essential",
            "curl",
            "sudo",
            "git",
            "git-lfs",
        )
        # Layer 3: Install uv (rarely changes)
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh > /uv-installer.sh",
            "INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR='/usr/local/bin' sh /uv-installer.sh",
            "rm /uv-installer.sh"
        )
        # Layer 4: Install prime CLI (rarely changes)
        .run_commands("uv tool install prime")
        # Add prime CLI to PATH for subsequent build steps
        .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
        .workdir("/app")
        # Layer 5: Install Python dependencies (only rebuilds when lockfiles change)
        # We copy only pyproject.toml, uv.lock, and README.md first, then run uv sync.
        # README.md is required because pyproject.toml references it and hatchling validates it.
        # This ensures code changes don't invalidate the dependency cache.
        .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
        .add_local_file("uv.lock", remote_path="/app/uv.lock", copy=True)
        .add_local_file("README.md", remote_path="/app/README.md", copy=True)
        .env({
            "UV_COMPILE_BYTECODE": "1",
            "UV_LINK_MODE": "copy",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "$PATH:/usr/local/cuda/bin",
        })
        .run_commands(
            "uv sync --no-dev --frozen --no-install-project",
            gpu="any"  # Some packages (flash-attn, vllm) require GPU during build
        )
        # Layer 6: Install environments from Prime Intellect hub
        .run_commands("prime env install primeintellect/single-turn-math --with uv")
        # Layer 7: Set runtime environment variables
        # Must be before add_local_* with copy=False (those are runtime mounts, not build steps)
        # Note: Modal .env() doesn't support shell variable expansion, so we set full paths
        .env({
            "PYTHONUNBUFFERED": "1",
            "PATH": "/app/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/app/src",
        })
        # Layers 8-12: Mount all project files at RUNTIME (not built into image)
        # Using copy=False means changes deploy instantly without image rebuilds!
        # These MUST be last since no build steps can follow copy=False
        .add_local_dir(
            "configs",
            remote_path="/app/configs",
            copy=False,
        )
        .add_local_dir(
            "src/prime_rl",
            remote_path="/app/src/prime_rl",
            copy=False,
        )
        .add_local_dir(
            "tests",
            remote_path="/app/tests",
            copy=False,
        )
        .add_local_dir(
            "examples",
            remote_path="/app/examples",
            copy=False,
        )
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
def run_command(
    cmd: str,
    output_dir: str | None = None,
    commit_volume: bool = True,
    commit_interval: int = 2400,
):
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
        output_dir: Output directory for RL commands (default: /data/outputs/<timestamp>). Only used if cmd is an RL command.
        commit_volume: Whether to commit volume periodically during execution (default: True). Set False for quick commands.
        commit_interval: Interval in seconds between volume commits (default: 2400 = 40 minutes).
    """
    import subprocess
    import sys
    import os
    import threading
    import time
    
    # Detect if this is an RL command
    is_rl_command = "uv run rl" in cmd or "rl @" in cmd

    # Append output-dir for RL commands if not present
    # Use timestamp to avoid overwriting previous runs
    if is_rl_command and output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"/data/outputs/{timestamp}"
    
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