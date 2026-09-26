from collections import defaultdict


def parse_matched_entity_ids(value):
    """
    Convert the comma-separated matched_entity_ids field
    into a set of entity IDs.

    Empty values represent no match.
    """

    if value is None:
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        entity_id.strip()
        for entity_id in value.split(",")
        if entity_id.strip()
    }


def build_ground_truth_map(ground_truth_df):
    """
    Build a mapping:

        Source 1 ID -> set of matching Source 2/Source 3 IDs
    """

    ground_truth = defaultdict(set)

    for _, row in ground_truth_df.iterrows():
        source1_id = str(row["source1_entity_id"]).strip()

        matches = parse_matched_entity_ids(
            row["matched_entity_ids"]
        )

        ground_truth[source1_id].update(matches)

    return dict(ground_truth)


def is_match(source1_id, candidate_id, ground_truth_map):
    """
    Return True if candidate_id is a known match
    for source1_id.
    """

    return candidate_id in ground_truth_map.get(source1_id, set())


def get_matches(source1_id, ground_truth_map):
    """
    Return all known matches for a Source 1 entity.
    """

    return ground_truth_map.get(source1_id, set())


def create_positive_pairs(ground_truth_map):
    """
    Convert the ground-truth mapping into a list of
    positive (Source 1, matched entity) pairs.
    """

    pairs = []

    for source1_id, matched_ids in ground_truth_map.items():
        for matched_id in matched_ids:
            pairs.append((source1_id, matched_id))

    return pairs