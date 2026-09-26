from pathlib import Path
from typing import Dict, Set

import pandas as pd


def write_matching_results(
    source1: pd.DataFrame,
    predictions: Dict[str, Set[str]],
    output_path: Path,
):
    """
    Write matching_results.tsv.
    """

    rows = []

    for s1_id in source1["entity_id"]:

        matched_ids = predictions.get(
            s1_id,
            set(),
        )

        matched_ids = sorted(
            matched_ids
        )

        rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(
                    matched_ids
                ),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def write_candidate_pairs(
    source1: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    output_path: Path,
):
    """
    Write candidate_pairs.tsv.
    """

    rows = []

    for s1_id in source1["entity_id"]:

        candidate_ids = candidates.get(
            s1_id,
            set(),
        )

        candidate_ids = sorted(
            candidate_ids
        )

        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    candidate_ids
                ),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )