import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from data_loader import load_entity_file, load_ground_truth
from normalization import normalize_entity_dataframe
from blocking import build_source_index
from features import create_pair_features
from ground_truth import build_ground_truth_map


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "matching_model.joblib"
)

FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "address_ratio",
    "address_partial_ratio",
    "address_token_jaccard",
    "country_match",
]

RANDOM_STATE = 42

# Process Source 1 in chunks.
SOURCE1_CHUNK_SIZE = 10_000

# Maximum number of negative examples retained per chunk.
MAX_NEGATIVES_PER_CHUNK = 50_000

# Maximum positive examples retained per chunk.
MAX_POSITIVES_PER_CHUNK = 50_000


def create_candidate_ids(
    source1_row,
    source2_df,
    source3_df,
    source2_index,
    source3_index,
):
    """
    Generate candidate IDs for one Source 1 entity.

    Candidates are generated from shared normalized
    name/address tokens.

    Country is used as a blocking filter only when
    both countries are present.
    """

    candidates = set()

    source1_country = source1_row.get(
        "country_normalized",
        "",
    )

    for index_name, index in source2_index.items():

        if index_name == "name":
            column = "name_normalized"
        else:
            column = "address_normalized"

        value = source1_row.get(column, "")

        if not value:
            continue

        for token in value.split():

            for row_index in index.get(token, set()):

                candidate_row = source2_df.iloc[row_index]

                candidate_country = candidate_row.get(
                    "country_normalized",
                    "",
                )

                if (
                    source1_country
                    and candidate_country
                    and source1_country != candidate_country
                ):
                    continue

                candidates.add(
                    candidate_row["entity_id"]
                )

    for index_name, index in source3_index.items():

        if index_name == "name":
            column = "name_normalized"
        else:
            column = "address_normalized"

        value = source1_row.get(column, "")

        if not value:
            continue

        for token in value.split():

            for row_index in index.get(token, set()):

                candidate_row = source3_df.iloc[row_index]

                candidate_country = candidate_row.get(
                    "country_normalized",
                    "",
                )

                if (
                    source1_country
                    and candidate_country
                    and source1_country != candidate_country
                ):
                    continue

                candidates.add(
                    candidate_row["entity_id"]
                )

    return candidates


def build_candidate_lookup(
    source2_df,
    source3_df,
):
    """
    Build entity ID -> row lookups.
    """

    source2_lookup = source2_df.set_index(
        "entity_id",
        drop=False,
    )

    source3_lookup = source3_df.set_index(
        "entity_id",
        drop=False,
    )

    return source2_lookup, source3_lookup


def create_training_rows(
    source1_chunk,
    source2_df,
    source3_df,
    source2_lookup,
    source3_lookup,
    source2_index,
    source3_index,
    ground_truth_map,
):
    """
    Create positive and negative feature rows
    for one Source 1 chunk.
    """

    positive_rows = []
    negative_rows = []

    for _, source1_row in source1_chunk.iterrows():

        source1_id = source1_row["entity_id"]

        true_matches = ground_truth_map.get(
            source1_id,
            set(),
        )

        candidate_ids = create_candidate_ids(
            source1_row,
            source2_df,
            source3_df,
            source2_index,
            source3_index,
        )

        if not candidate_ids:
            continue

        for candidate_id in candidate_ids:

            if candidate_id.startswith("S2-"):

                if candidate_id not in source2_lookup.index:
                    continue

                candidate_row = source2_lookup.loc[
                    candidate_id
                ]

            elif candidate_id.startswith("S3-"):

                if candidate_id not in source3_lookup.index:
                    continue

                candidate_row = source3_lookup.loc[
                    candidate_id
                ]

            else:
                continue

            features = create_pair_features(
                source1_row,
                candidate_row,
            )

            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = candidate_id

            if candidate_id in true_matches:
                features["label"] = 1
                positive_rows.append(features)
            else:
                features["label"] = 0
                negative_rows.append(features)

    return positive_rows, negative_rows


def sample_rows(
    rows,
    maximum,
):
    """
    Randomly sample rows when a chunk produces
    too many examples.
    """

    if len(rows) <= maximum:
        return rows

    random.seed(RANDOM_STATE)

    return random.sample(
        rows,
        maximum,
    )


def prepare_reference_data():
    """
    Load Source 2 and Source 3 once.

    Source 1 is intentionally NOT loaded entirely.
    """

    print("Loading Source 2...")

    source2 = load_entity_file(
        TRAIN_SOURCE2
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    print("Loading Source 3...")

    source3 = load_entity_file(
        TRAIN_SOURCE3
    )

    print(
        f"Source 3 rows: {len(source3):,}"
    )

    print("Normalizing Source 2 and Source 3...")

    source2 = normalize_entity_dataframe(
        source2
    )

    source3 = normalize_entity_dataframe(
        source3
    )

    print("Building Source 2 index...")

    source2_index = build_source_index(
        source2
    )

    print("Building Source 3 index...")

    source3_index = build_source_index(
        source3
    )

    source2_lookup, source3_lookup = (
        build_candidate_lookup(
            source2,
            source3,
        )
    )

    return (
        source2,
        source3,
        source2_lookup,
        source3_lookup,
        source2_index,
        source3_index,
    )


def prepare_ground_truth():
    print("Loading ground truth...")

    ground_truth = load_ground_truth(
        TRAIN_GROUND_TRUTH
    )

    print(
        f"Ground-truth rows: {len(ground_truth):,}"
    )

    return build_ground_truth_map(
        ground_truth
    )


def build_training_dataset():
    """
    Process Source 1 in chunks and construct a
    manageable training dataset.
    """

    (
        source2,
        source3,
        source2_lookup,
        source3_lookup,
        source2_index,
        source3_index,
    ) = prepare_reference_data()

    ground_truth_map = prepare_ground_truth()

    positive_rows = []
    negative_rows = []

    print()
    print("Processing Source 1 in chunks...")

    reader = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        chunksize=SOURCE1_CHUNK_SIZE,
    )

    processed_rows = 0

    for chunk_number, source1_chunk in enumerate(
        reader,
        start=1,
    ):

        source1_chunk = source1_chunk[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        ].copy()

        source1_chunk = normalize_entity_dataframe(
            source1_chunk
        )

        chunk_positive, chunk_negative = (
            create_training_rows(
                source1_chunk,
                source2,
                source3,
                source2_lookup,
                source3_lookup,
                source2_index,
                source3_index,
                ground_truth_map,
            )
        )

        positive_rows.extend(
            sample_rows(
                chunk_positive,
                MAX_POSITIVES_PER_CHUNK,
            )
        )

        negative_rows.extend(
            sample_rows(
                chunk_negative,
                MAX_NEGATIVES_PER_CHUNK,
            )
        )

        processed_rows += len(
            source1_chunk
        )

        print(
            f"Processed Source 1 rows: "
            f"{processed_rows:,}"
        )

        # Stop after enough examples have been collected.
        #
        # This gives us a manageable initial model
        # training set instead of creating millions
        # of feature rows.
        if (
            len(positive_rows) >= 100_000
            and len(negative_rows) >= 300_000
        ):
            print(
                "Enough training examples collected."
            )
            break

    positive_rows = sample_rows(
        positive_rows,
        100_000,
    )

    negative_rows = sample_rows(
        negative_rows,
        300_000,
    )

    rows = positive_rows + negative_rows

    random.seed(RANDOM_STATE)
    random.shuffle(rows)

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        raise ValueError(
            "No training examples were generated."
        )

    print()
    print(
        f"Final positive examples: "
        f"{sum(feature_df['label'] == 1):,}"
    )

    print(
        f"Final negative examples: "
        f"{sum(feature_df['label'] == 0):,}"
    )

    print(
        f"Final training examples: "
        f"{len(feature_df):,}"
    )

    return feature_df


def train_model(feature_df):
    X = feature_df[
        FEATURE_COLUMNS
    ]

    y = feature_df["label"]

    print()
    print("Training RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


def main():
    print("=" * 60)
    print("ENTITY MATCHING MODEL TRAINING")
    print("=" * 60)

    feature_df = build_training_dataset()

    model = train_model(
        feature_df
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    print()
    print("Model saved successfully:")
    print(MODEL_PATH)

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()