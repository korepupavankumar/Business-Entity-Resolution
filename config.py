from pathlib import Path


# ============================================================
# Project directories
# ============================================================

# Project root:
# Amazon ML Challenge/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"


# ============================================================
# Training files
# ============================================================

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# Test files
# ============================================================

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# ============================================================
# Output files
# ============================================================

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# Dataset columns
# ============================================================

ENTITY_ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GROUND_TRUTH_SOURCE1_COL = "source1_entity_id"
GROUND_TRUTH_MATCHES_COL = "matched_entity_ids"


# ============================================================
# General settings
# ============================================================

RANDOM_STATE = 42

# Number of rows processed at a time when reading large TSV files.
CHUNK_SIZE = 100_000


# Make sure the output directory exists.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)