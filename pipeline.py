import pandas as pd

from .config import (
    DEFAULT_MATCH_THRESHOLD,
    MODEL_FILE,
    MATCHING_RESULTS,
    CANDIDATE_PAIRS,
)

from .data_loader import (
    load_training_data,
    load_test_data,
)

from .normalization import (
    prepare_all_sources,
)

from .blocking import (
    generate_all_candidates,
)

from .ground_truth import (
    build_ground_truth_map,
)

from .train_model import (
    create_training_pairs,
    train_classifier,
    save_model,
)

from .predict import (
    predict_candidates,
)

from .output_writer import (
    write_matching_results,
    write_candidate_pairs,
)

from .evaluate import (
    blocking_recall,
    macro_f05,
)


def run_training():
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    print(
        f"Source 1 training rows: {len(source1)}"
    )

    print(
        f"Source 2 training rows: {len(source2)}"
    )

    print(
        f"Source 3 training rows: {len(source3)}"
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    source1, source2, source3 = (
        prepare_all_sources(
            source1,
            source2,
            source3,
        )
    )

    # -----------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------

    candidates = generate_all_candidates(
        source1,
        source2,
        source3,
    )

    ground_truth_map = build_ground_truth_map(
        ground_truth
    )

    recall = blocking_recall(
        candidates,
        ground_truth_map,
    )

    print(
        f"Training blocking recall: {recall:.4f}"
    )

    # -----------------------------------------------------
    # Build pair dataset
    # -----------------------------------------------------

    pair_df = create_training_pairs(
        source1,
        source2,
        source3,
        ground_truth,
    )

    print(
        f"Training pair count: {len(pair_df)}"
    )

    print(
        "Positive pairs:",
        int(pair_df["label"].sum()),
    )

    print(
        "Negative pairs:",
        int((pair_df["label"] == 0).sum()),
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    model = train_classifier(
        pair_df
    )

    save_model(
        model,
        MODEL_FILE,
    )

    print(
        f"Model saved to: {MODEL_FILE}"
    )

    return model


def run_prediction(
    threshold: float = DEFAULT_MATCH_THRESHOLD,
):
    print("=" * 70)
    print("TEST PREDICTION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load test
    # -----------------------------------------------------

    source1, source2, source3 = (
        load_test_data()
    )

    print(
        f"Test Source 1 rows: {len(source1)}"
    )

    print(
        f"Test Source 2 rows: {len(source2)}"
    )

    print(
        f"Test Source 3 rows: {len(source3)}"
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    source1, source2, source3 = (
        prepare_all_sources(
            source1,
            source2,
            source3,
        )
    )

    # -----------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------

    candidates = generate_all_candidates(
        source1,
        source2,
        source3,
    )

    total_candidates = sum(
        len(x)
        for x in candidates.values()
    )

    print(
        f"Total candidate pairs: {total_candidates}"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    from .train_model import load_model

    model = load_model(
        MODEL_FILE
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = predict_candidates(
        source1,
        source2,
        source3,
        candidates,
        model,
        threshold=threshold,
    )

    # -----------------------------------------------------
    # Write candidate file
    # -----------------------------------------------------

    write_candidate_pairs(
        source1,
        candidates,
        CANDIDATE_PAIRS,
    )

    # -----------------------------------------------------
    # Write matching file
    # -----------------------------------------------------

    write_matching_results(
        source1,
        predictions,
        MATCHING_RESULTS,
    )

    matched_entities = sum(
        bool(v)
        for v in predictions.values()
    )

    print(
        f"Source 1 entities with >=1 match: "
        f"{matched_entities}"
    )

    print(
        f"Candidate output: {CANDIDATE_PAIRS}"
    )

    print(
        f"Matching output: {MATCHING_RESULTS}"
    )


def main():
    run_training()
    run_prediction()


if __name__ == "__main__":
    main()