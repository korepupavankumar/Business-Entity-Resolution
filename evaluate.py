from typing import Dict, Set


def f05_score(
    precision: float,
    recall: float,
) -> float:

    beta = 0.5

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denominator
    )


def entity_precision_recall(
    predicted: Set[str],
    actual: Set[str],
):
    """Calculate precision/recall for one Source 1 entity."""

    predicted = set(predicted)
    actual = set(actual)

    if not predicted and not actual:
        return 1.0, 1.0

    if not predicted and actual:
        return 1.0, 0.0

    if predicted and not actual:
        return 0.0, 1.0

    true_positive = len(
        predicted & actual
    )

    precision = (
        true_positive / len(predicted)
    )

    recall = (
        true_positive / len(actual)
    )

    return precision, recall


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
):
    """
    Calculate macro F0.5 across Source 1 entities.
    """

    scores = []

    for s1_id, actual in ground_truth.items():

        predicted = predictions.get(
            s1_id,
            set(),
        )

        precision, recall = (
            entity_precision_recall(
                predicted,
                actual,
            )
        )

        score = f05_score(
            precision,
            recall,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
):
    """
    Calculate how many true matches survived blocking.
    """

    total_true = 0
    recovered_true = 0

    for s1_id, actual_ids in ground_truth.items():

        candidate_ids = candidates.get(
            s1_id,
            set(),
        )

        total_true += len(actual_ids)

        recovered_true += len(
            actual_ids & candidate_ids
        )

    if total_true == 0:
        return 0.0

    return recovered_true / total_true