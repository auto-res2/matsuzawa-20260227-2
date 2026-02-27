"""Main entry point for orchestrating experiment runs."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> int:
    """
    Orchestrate a single experiment run.

    This is the main entry point that:
    1. Loads the Hydra config
    2. Applies mode overrides
    3. Invokes inference module directly

    Args:
        cfg: Hydra configuration

    Returns:
        Exit code (0 for success)
    """
    print("\n" + "=" * 80)
    print(f"Main Orchestrator: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Determine task type from config
    task_type = "inference"  # This experiment is inference-only

    print(f"\nTask type: {task_type}")
    print(f"Method: {cfg.run.method.name} ({cfg.run.method.type})")
    print(f"Dataset: {cfg.dataset.name} ({cfg.dataset.max_samples} samples)")
    print(f"Model: {cfg.model.name}")

    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"\nConfig saved to: {config_path}")

    # Import and run inference directly
    from inference import main as inference_main

    print(f"\nInvoking inference module...")

    # Execute inference
    try:
        return_code = inference_main(cfg)
    except Exception as e:
        print(f"\n[ERROR] Inference failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if return_code != 0:
        print(f"\n[ERROR] Inference failed with exit code: {return_code}")
        return return_code

    print(f"\n[SUCCESS] Run completed: {cfg.run.run_id}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
