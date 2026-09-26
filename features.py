import re
import pandas as pd
from rapidfuzz import fuzz


def tokenize(text):
    if not text:
        return set()

    return {
        token
        for token in re.split(r"\s+", str(text).strip())
        if token
    }


def token_jaccard(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union else 0.0


def string_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0

    return fuzz.ratio(str(text1), str(text2)) / 100.0


def partial_string_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0

    return fuzz.partial_ratio(str(text1), str(text2)) / 100.0


def country_match(country1, country2):
    if not country1 or not country2:
        return 0.0

    return float(
        str(country1).strip().lower()
        == str(country2).strip().lower()
    )


def create_pair_features(source1_row, candidate_row):
    """
    Create numerical similarity features for one Source 1 /
    candidate Source 2 or Source 3 pair.
    """

    name1 = source1_row.get("name_normalized", "")
    name2 = candidate_row.get("name_normalized", "")

    address1 = source1_row.get("address_normalized", "")
    address2 = candidate_row.get("address_normalized", "")

    country1 = source1_row.get("country_normalized", "")
    country2 = candidate_row.get("country_normalized", "")

    return {
        "name_ratio": string_similarity(name1, name2),
        "name_partial_ratio": partial_string_similarity(name1, name2),
        "name_token_jaccard": token_jaccard(name1, name2),

        "address_ratio": string_similarity(address1, address2),
        "address_partial_ratio": partial_string_similarity(
            address1,
            address2,
        ),
        "address_token_jaccard": token_jaccard(
            address1,
            address2,
        ),

        "country_match": country_match(
            country1,
            country2,
        ),
    }


def create_feature_dataframe(
    source1_df,
    source2_df,
    source3_df,
    candidate_pairs,
):
    """
    Convert candidate pairs into a feature dataframe.

    candidate_pairs format:

        {
            "S1-123": {"S2-456", "S3-789"},
            "S1-999": {"S2-111"},
        }

    Each output row represents one Source 1 -> candidate pair.
    """

    source1_lookup = source1_df.set_index(
        "entity_id",
        drop=False,
    )

    source2_lookup = source2_df.set_index(
        "entity_id",
        drop=False,
    )

    source3_lookup = source3_df.set_index(
        "entity_id",
        drop=False,
    )

    rows = []

    for source1_id, candidate_ids in candidate_pairs.items():

        if source1_id not in source1_lookup.index:
            continue

        source1_row = source1_lookup.loc[source1_id]

        for candidate_id in candidate_ids:

            if candidate_id.startswith("S2-"):
                if candidate_id not in source2_lookup.index:
                    continue

                candidate_row = source2_lookup.loc[candidate_id]

            elif candidate_id.startswith("S3-"):
                if candidate_id not in source3_lookup.index:
                    continue

                candidate_row = source3_lookup.loc[candidate_id]

            else:
                continue

            features = create_pair_features(
                source1_row,
                candidate_row,
            )

            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = candidate_id

            rows.append(features)

    columns = [
        "source1_entity_id",
        "candidate_entity_id",

        "name_ratio",
        "name_partial_ratio",
        "name_token_jaccard",

        "address_ratio",
        "address_partial_ratio",
        "address_token_jaccard",

        "country_match",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)