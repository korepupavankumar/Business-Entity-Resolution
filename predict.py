from typing import Dict, Set

import pandas as pd

from .features import (
    create_pair_features,
    FEATURE_COLUMNS,
)


def predict_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    model,
    threshold: float = 0.70,
):
    """
    Score every candidate pair and return final matches.
    """

    target_lookup = {}

    for _, row in source2.iterrows():
        target_lookup[row["entity_id"]] = row

    for _, row in source3.iterrows():
        target_lookup[row["entity_id"]] = row

    source1_lookup = {
        row["entity_id"]: row
        for _, row in source1.iterrows()
    }

    predictions = {}

    for s1_id, candidate_ids in candidates.items():

        s1_row = source1_lookup[s1_id]

        scored = []

        for candidate_id in candidate_ids:

            target_row = target_lookup.get(
                candidate_id
            )

            if target_row is None:
                continue

            features = create_pair_features(
                s1_row,
                target_row,
            )

            feature_df = pd.DataFrame(
                [features]
            )[FEATURE_COLUMNS]

            probability = model.predict_proba(
                feature_df
            )[0, 1]

            scored.append(
                (
                    probability,
                    candidate_id,
                )
            )

        # -------------------------------------------------
        # Select candidates over threshold
        # -------------------------------------------------

        selected = [
            candidate_id
            for probability, candidate_id
            in scored
            if probability >= threshold
        ]

        # -------------------------------------------------
        # Conservative singleton logic
        # -------------------------------------------------

        # If no candidate clears the threshold, return
        # an empty set.
        predictions[s1_id] = set(selected)

    return predictions