"""Inference script for prompt-based experiments."""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

from preprocess import load_gsm8k, check_answer_correct, normalize_answer


def run_baseline_inference(
    cfg: DictConfig,
    samples: List[Dict[str, Any]],
    llm: LLM,
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """
    Run baseline single-call Chain-of-Thought inference.

    Args:
        cfg: Hydra config
        samples: List of dataset samples
        llm: vLLM model
        sampling_params: Sampling parameters

    Returns:
        List of results with predictions
    """
    print(f"\n=== Running Baseline CoT Inference ({len(samples)} samples) ===")

    prompt_template = cfg.run.inference.prompt_template
    results = []

    # Prepare prompts
    prompts = [prompt_template.format(question=s["question"]) for s in samples]

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    # Process outputs
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        response_text = output.outputs[0].text

        # Extract answer
        try:
            predicted = normalize_answer(response_text)
            is_correct = check_answer_correct(response_text, sample["answer"])
        except (ValueError, TypeError):
            predicted = None
            is_correct = False

        result = {
            "idx": sample["idx"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "response": response_text,
            "predicted": predicted,
            "correct": is_correct,
            "abstained": False,
            "num_calls": 1,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples")

    return results


def run_ebc3ot_inference(
    cfg: DictConfig,
    samples: List[Dict[str, Any]],
    llm: LLM,
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """
    Run Evidence-Budgeted Commit-Check CoT inference (two calls).

    Args:
        cfg: Hydra config
        samples: List of dataset samples
        llm: vLLM model
        sampling_params: Sampling parameters

    Returns:
        List of results with predictions
    """
    print(f"\n=== Running EB-C3oT Inference ({len(samples)} samples) ===")

    commit_template = cfg.run.inference.commit_prompt_template
    check_template = cfg.run.inference.check_prompt_template
    revision_budget = cfg.run.method.revision_budget
    risk_threshold = cfg.run.method.risk_threshold

    results = []

    for i, sample in enumerate(samples):
        # Commit pass: Generate structured claims
        commit_prompt = commit_template.format(question=sample["question"])
        commit_output = llm.generate([commit_prompt], sampling_params)[0]
        commit_response = commit_output.outputs[0].text

        # Check pass: Verify claims and apply decision rule
        check_prompt = check_template.format(
            question=sample["question"],
            commit_response=commit_response,
            revision_budget=revision_budget,
            risk_threshold=risk_threshold,
        )
        check_output = llm.generate([check_prompt], sampling_params)[0]
        check_response = check_output.outputs[0].text

        # Parse check response for decision
        abstained = "ABSTAIN" in check_response

        # Extract answer
        try:
            if not abstained:
                # Look for FINAL ANSWER in check response
                answer_match = re.search(
                    r"FINAL ANSWER[:\s]+([-+]?[\d,]+\.?\d*)", check_response
                )
                if answer_match:
                    predicted_str = answer_match.group(1)
                else:
                    # Fallback to commit response
                    predicted_str = commit_response

                predicted = normalize_answer(predicted_str)
                is_correct = check_answer_correct(predicted_str, sample["answer"])
            else:
                predicted = None
                is_correct = False
        except (ValueError, TypeError):
            predicted = None
            is_correct = False
            abstained = True

        # Extract broken count and risk score
        broken_count = extract_metric(check_response, "BROKEN_COUNT")
        risk_score = extract_metric(check_response, "RISK_SCORE")

        result = {
            "idx": sample["idx"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "commit_response": commit_response,
            "check_response": check_response,
            "predicted": predicted,
            "correct": is_correct,
            "abstained": abstained,
            "broken_count": broken_count,
            "risk_score": risk_score,
            "num_calls": 2,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples")

    return results


def extract_metric(text: str, metric_name: str) -> Optional[float]:
    """Extract a numeric metric from check response."""
    pattern = rf"{metric_name}[:\s]+([-+]?[\d.]+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute evaluation metrics from results.

    Args:
        results: List of inference results

    Returns:
        Dict of metric name -> value
    """
    total = len(results)
    answered = [r for r in results if not r["abstained"]]
    num_answered = len(answered)
    num_abstained = total - num_answered

    num_correct = sum(1 for r in answered if r["correct"])

    # Core metrics
    metrics = {
        "total_samples": total,
        "num_answered": num_answered,
        "num_abstained": num_abstained,
        "abstain_rate": num_abstained / total if total > 0 else 0,
        "accuracy_all": num_correct / total if total > 0 else 0,
        "accuracy_answered": num_correct / num_answered if num_answered > 0 else 0,
        "total_correct": num_correct,
    }

    # Average broken count and risk score (for EB-C3oT)
    if "broken_count" in results[0]:
        broken_counts = [
            r.get("broken_count", 0)
            for r in results
            if r.get("broken_count") is not None
        ]
        risk_scores = [
            r.get("risk_score", 0) for r in results if r.get("risk_score") is not None
        ]

        if broken_counts:
            metrics["avg_broken_count"] = sum(broken_counts) / len(broken_counts)
        if risk_scores:
            metrics["avg_risk_score"] = sum(risk_scores) / len(risk_scores)

    return metrics


def main(cfg: DictConfig):
    """Main inference entry point."""
    print("\n" + "=" * 80)
    print(f"Starting inference for run: {cfg.run.run_id}")
    print("=" * 80)

    # Apply mode overrides
    if cfg.mode == "sanity_check":
        cfg.dataset.max_samples = 10
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print("\n[SANITY CHECK MODE] Reducing to 10 samples")

    # Initialize WandB
    if cfg.wandb.mode == "online":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"\nWandB run: {wandb.run.url}")

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: vLLM initialization failed with FileNotFoundError: [Errno 2] No such file or directory: ''
    # [CAUSE]: cfg.cache_dir is a relative path (".cache"), but vLLM's download_dir needs an absolute path.
    #          When Hydra changes working directory, relative paths can resolve incorrectly or to empty strings.
    # [FIX]: Convert cache_dir to absolute path before passing to vLLM and dataset loader.
    #
    # [OLD CODE]:
    # samples = load_gsm8k(
    #     split=cfg.dataset.split,
    #     max_samples=cfg.dataset.max_samples,
    #     cache_dir=cfg.cache_dir,
    # )
    # llm = LLM(
    #     model=cfg.model.name,
    #     download_dir=cfg.cache_dir,
    #     tensor_parallel_size=1,
    # )
    #
    # [NEW CODE]:
    # Resolve cache_dir to absolute path
    cache_dir_abs = Path(cfg.cache_dir).resolve()
    cache_dir_abs.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir_abs)
    print(f"\nCache directory: {cache_dir_str}")

    # Load dataset
    samples = load_gsm8k(
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
        cache_dir=cache_dir_str,
    )

    # Initialize model
    print(f"\nLoading model: {cfg.model.name}")
    llm = LLM(
        model=cfg.model.name,
        download_dir=cache_dir_str,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens, top_p=1.0
    )

    # Run inference based on method type
    if cfg.run.method.type == "baseline":
        results = run_baseline_inference(cfg, samples, llm, sampling_params)
    elif cfg.run.method.type == "proposed":
        results = run_ebc3ot_inference(cfg, samples, llm, sampling_params)
    else:
        raise ValueError(f"Unknown method type: {cfg.run.method.type}")

    # Compute metrics
    metrics = compute_metrics(results)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("=" * 80)

    # Log to WandB
    if cfg.wandb.mode == "online":
        wandb.log(metrics)
        for key, value in metrics.items():
            wandb.summary[key] = value
        wandb.finish()

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(results, metrics, cfg)

    return 0


def perform_sanity_validation(
    results: List[Dict[str, Any]], metrics: Dict[str, float], cfg: DictConfig
):
    """Perform sanity validation checks."""
    print("\n" + "=" * 80)
    print("SANITY VALIDATION")
    print("=" * 80)

    # Check: at least 5 samples processed
    samples_processed = metrics["total_samples"]

    # Check: all metrics are finite
    all_finite = all(
        isinstance(v, (int, float)) and not (v != v)  # not NaN
        for v in metrics.values()
        if isinstance(v, (int, float))
    )

    # Check: outputs are valid (at least some answered or all abstained intentionally)
    valid_outputs = (
        metrics["num_answered"] > 0
        or metrics["num_abstained"] == metrics["total_samples"]
    )

    # Summary
    summary = {
        "samples": samples_processed,
        "answered": metrics["num_answered"],
        "abstained": metrics["num_abstained"],
        "correct": metrics.get("total_correct", 0),
        "accuracy_answered": metrics.get("accuracy_answered", 0),
    }

    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Validation result
    if samples_processed >= 5 and all_finite and valid_outputs:
        print("SANITY_VALIDATION: PASS")
    else:
        reasons = []
        if samples_processed < 5:
            reasons.append("insufficient_samples")
        if not all_finite:
            reasons.append("invalid_metrics")
        if not valid_outputs:
            reasons.append("invalid_outputs")
        print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")


if __name__ == "__main__":
    import hydra
    from hydra import compose, initialize_config_dir

    # Parse command-line arguments
    config_dir = Path(__file__).parent.parent / "config"

    with initialize_config_dir(
        config_dir=str(config_dir.absolute()), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=sys.argv[1:])

    sys.exit(main(cfg))
