import pandas as pd


REQUIRED_ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

REQUIRED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def load_entity_file(path):
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    missing = [
        column
        for column in REQUIRED_ENTITY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    return df[REQUIRED_ENTITY_COLUMNS].copy()


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    missing = [
        column
        for column in REQUIRED_GROUND_TRUTH_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    return df[REQUIRED_GROUND_TRUTH_COLUMNS].copy()


def load_training_data(
    source1_path,
    source2_path,
    source3_path,
    ground_truth_path,
):
    source1 = load_entity_file(source1_path)
    source2 = load_entity_file(source2_path)
    source3 = load_entity_file(source3_path)
    ground_truth = load_ground_truth(ground_truth_path)

    return (
        source1,
        source2,
        source3,
        ground_truth,
    )


def load_test_data(
    source1_path,
    source2_path,
    source3_path,
):
    source1 = load_entity_file(source1_path)
    source2 = load_entity_file(source2_path)
    source3 = load_entity_file(source3_path)

    return source1, source2, source3