"""Dataset preprocessing utilities for GSM8K and other datasets."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ('train' or 'test')
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Directory to cache the dataset

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    print(f"Loading GSM8K dataset (split={split}, max_samples={max_samples})...")

    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts and extract numeric answer
    samples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer from answer text (format: "#### number")
        numeric_answer = extract_numeric_answer(answer_text)

        samples.append(
            {
                "idx": i,
                "question": question,
                "answer": numeric_answer,
                "answer_text": answer_text,
            }
        )

    print(f"Loaded {len(samples)} samples from GSM8K {split} split")
    return samples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer text.

    GSM8K answers are in format: "<reasoning>\n#### <number>"

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numeric answer as float
    """
    # Look for "####" marker
    match = re.search(r"####\s*([-+]?[\d,]+\.?\d*)", answer_text)
    if match:
        # Remove commas and convert to float
        numeric_str = match.group(1).replace(",", "")
        return float(numeric_str)

    # Fallback: try to find any number at the end
    numbers = re.findall(r"[-+]?[\d,]+\.?\d*", answer_text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    raise ValueError(f"Could not extract numeric answer from: {answer_text}")


def normalize_answer(answer: str) -> float:
    """
    Normalize a predicted answer string to numeric value.

    Args:
        answer: Predicted answer string

    Returns:
        Normalized numeric value
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: All predictions were 1.0 even though model responses contained correct answers like 18, 3, 70000, etc.
    # [CAUSE]: The regex was extracting the first number in the text, which was often "1" from "Step 1:" in the reasoning.
    #          The model outputs answers in LaTeX format like "$\boxed{18}$" or "The final answer is: $\boxed{18}$"
    #          but the function was not prioritizing these answer markers.
    # [FIX]: First try to extract from common answer patterns (boxed, "final answer is", etc.),
    #        then fall back to finding the last number in the text (more likely to be the answer than the first).
    #
    # [OLD CODE]:
    # # Remove common text patterns
    # answer = answer.lower()
    # answer = re.sub(r"(the answer is|final answer|answer:)", "", answer)
    # answer = answer.strip()
    #
    # # Extract first number found
    # numbers = re.findall(r"[-+]?[\d,]+\.?\d*", answer)
    # if numbers:
    #     return float(numbers[0].replace(",", ""))
    #
    # [NEW CODE]:

    # Strategy 1: Try to extract from \boxed{} LaTeX format (most reliable)
    boxed_match = re.search(r"\\boxed\{([-+]?[\d,]+\.?\d*)\}", answer)
    if boxed_match:
        return float(boxed_match.group(1).replace(",", ""))

    # Strategy 2: Look for "final answer is" or similar patterns
    final_answer_patterns = [
        r"final answer is[:\s]*\$?\\?boxed\{?([-+]?[\d,]+\.?\d*)\}?",
        r"the answer is[:\s]*\$?\\?boxed\{?([-+]?[\d,]+\.?\d*)\}?",
        r"answer[:\s]+([-+]?[\d,]+\.?\d*)\s*$",
    ]

    answer_lower = answer.lower()
    for pattern in final_answer_patterns:
        match = re.search(pattern, answer_lower)
        if match:
            return float(match.group(1).replace(",", ""))

    # Strategy 3: Extract all numbers and return the LAST one (most likely the final answer)
    numbers = re.findall(r"[-+]?[\d,]+\.?\d*", answer)
    if numbers:
        # Filter out numbers that are likely step numbers (single digits 1-9 followed by colon or period)
        # and prefer larger numbers
        filtered = []
        for num_str in numbers:
            num_val = float(num_str.replace(",", ""))
            # Skip obvious step numbers (single digit followed by nothing substantial)
            if num_val >= 10 or len(numbers) == 1:
                filtered.append(num_str)

        if filtered:
            return float(filtered[-1].replace(",", ""))
        # If all filtered out, just use the last number
        return float(numbers[-1].replace(",", ""))

    # If no number found, raise error
    raise ValueError(f"Could not extract numeric value from: {answer}")


def check_answer_correct(
    predicted: str, ground_truth: float, tolerance: float = 1e-3
) -> bool:
    """
    Check if predicted answer matches ground truth.

    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth numeric value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if answers match within tolerance
    """
    try:
        predicted_num = normalize_answer(predicted)
        return abs(predicted_num - ground_truth) < tolerance
    except (ValueError, TypeError):
        return False
