import re
import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "blocking_index.db"
)

TOKEN_FREQUENCY_LIMIT = 200


def tokenize(text):
    if not text:
        return set()

    return {
        token
        for token in re.split(r"\s+", str(text).strip())
        if token
    }


def normalize_blocking_key(text):
    if not text:
        return ""

    return " ".join(str(text).strip().split())


def create_database():
    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        PRAGMA journal_mode=WAL
        """
    )

    connection.execute(
        """
        PRAGMA synchronous=NORMAL
        """
    )

    connection.execute(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT,
            name_normalized TEXT,
            address_normalized TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE token_counts (
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            country TEXT NOT NULL,
            token TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (
                source,
                field,
                country,
                token
            )
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE token_index (
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            country TEXT NOT NULL,
            token TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_token_index_lookup
        ON token_index (
            source,
            field,
            country,
            token
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE exact_index (
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            country TEXT NOT NULL,
            value TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_exact_lookup
        ON exact_index (
            source,
            field,
            country,
            value
        )
        """
    )

    return connection


def insert_entities(
    connection,
    file_path,
    source_name,
    chunk_size=100_000,
):
    print(
        f"Reading {source_name}: "
        f"{file_path}"
    )

    reader = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )

    total = 0

    for chunk in reader:

        rows = []

        for row in chunk.itertuples(index=False):

            entity_id = str(row.entity_id).strip()
            country = str(row.country).strip().lower()

            name = normalize_blocking_key(
                row.business_name
            )

            address = normalize_blocking_key(
                row.business_address
            )

            rows.append(
                (
                    entity_id,
                    source_name,
                    country,
                    name,
                    address,
                )
            )

        connection.executemany(
            """
            INSERT INTO entities (
                entity_id,
                source,
                country,
                name_normalized,
                address_normalized
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

        total += len(rows)

        print(
            f"  {source_name}: "
            f"{total:,} rows indexed"
        )

        connection.commit()


def build_exact_indexes(connection):

    print()
    print("Building exact blocking indexes...")

    connection.execute(
        """
        INSERT INTO exact_index (
            source,
            field,
            country,
            value,
            entity_id
        )
        SELECT
            source,
            'name',
            country,
            name_normalized,
            entity_id
        FROM entities
        WHERE name_normalized != ''
        """
    )

    connection.execute(
        """
        INSERT INTO exact_index (
            source,
            field,
            country,
            value,
            entity_id
        )
        SELECT
            source,
            'address',
            country,
            address_normalized,
            entity_id
        FROM entities
        WHERE address_normalized != ''
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_exact_name
        ON exact_index (
            source,
            field,
            country,
            value
        )
        """
    )

    connection.commit()


def build_token_counts(
    connection,
):

    print()
    print("Counting blocking tokens...")

    connection.execute(
        """
        CREATE TEMP TABLE entity_tokens (
            source TEXT,
            field TEXT,
            country TEXT,
            token TEXT
        )
        """
    )

    cursor = connection.execute(
        """
        SELECT
            entity_id,
            source,
            country,
            name_normalized,
            address_normalized
        FROM entities
        """
    )

    batch = []

    processed = 0

    while True:

        rows = cursor.fetchmany(10_000)

        if not rows:
            break

        for (
            entity_id,
            source,
            country,
            name,
            address,
        ) in rows:

            for token in tokenize(name):

                batch.append(
                    (
                        source,
                        "name",
                        country,
                        token,
                    )
                )

            for token in tokenize(address):

                batch.append(
                    (
                        source,
                        "address",
                        country,
                        token,
                    )
                )

        if len(batch) >= 100_000:

            connection.executemany(
                """
                INSERT INTO entity_tokens (
                    source,
                    field,
                    country,
                    token
                )
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

            batch.clear()

        processed += len(rows)

        if processed % 500_000 == 0:

            print(
                f"  processed: "
                f"{processed:,}"
            )

    if batch:

        connection.executemany(
            """
            INSERT INTO entity_tokens (
                source,
                field,
                country,
                token
            )
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )

    connection.commit()

    print("Aggregating token frequencies...")

    connection.execute(
        """
        INSERT INTO token_counts (
            source,
            field,
            country,
            token,
            count
        )
        SELECT
            source,
            field,
            country,
            token,
            COUNT(*)
        FROM entity_tokens
        GROUP BY
            source,
            field,
            country,
            token
        """
    )

    connection.commit()

    connection.execute(
        """
        DROP TABLE entity_tokens
        """
    )

    connection.commit()


def build_token_indexes(
    connection,
):

    print()
    print(
        "Building rare-token blocking indexes..."
    )

    connection.execute(
        """
        INSERT INTO token_index (
            source,
            field,
            country,
            token,
            entity_id
        )
        SELECT
            e.source,
            'name',
            e.country,
            t.token,
            e.entity_id
        FROM entities e
        JOIN token_counts t
          ON t.source = e.source
         AND t.country = e.country
         AND t.field = 'name'
        WHERE t.count <= ?
          AND t.token != ''
          AND (
              ' ' || e.name_normalized || ' '
          ) LIKE '% ' || t.token || ' %'
        """,
        (TOKEN_FREQUENCY_LIMIT,),
    )

    connection.execute(
        """
        INSERT INTO token_index (
            source,
            field,
            country,
            token,
            entity_id
        )
        SELECT
            e.source,
            'address',
            e.country,
            t.token,
            e.entity_id
        FROM entities e
        JOIN token_counts t
          ON t.source = e.source
         AND t.country = e.country
         AND t.field = 'address'
        WHERE t.count <= ?
          AND t.token != ''
          AND (
              ' ' || e.address_normalized || ' '
          ) LIKE '% ' || t.token || ' %'
        """,
        (TOKEN_FREQUENCY_LIMIT,),
    )

    connection.commit()

    print("Creating token index...")

    connection.execute(
        """
        CREATE INDEX idx_token_lookup
        ON token_index (
            source,
            field,
            country,
            token
        )
        """
    )

    connection.commit()


def build_blocking_database(
    source2_path,
    source3_path,
):
    print("=" * 60)
    print("BUILDING DISK-BACKED BLOCKING DATABASE")
    print("=" * 60)

    connection = create_database()

    try:

        insert_entities(
            connection,
            source2_path,
            "S2",
        )

        insert_entities(
            connection,
            source3_path,
            "S3",
        )

        build_exact_indexes(
            connection
        )

        build_token_counts(
            connection
        )

        build_token_indexes(
            connection
        )

        connection.execute(
            """
            ANALYZE
            """
        )

        connection.commit()

        print()
        print(
            "Blocking database created:"
        )
        print(DB_PATH)

    finally:

        connection.close()


def get_exact_candidates(
    connection,
    source,
    country,
    field,
    value,
):
    if not value:
        return set()

    rows = connection.execute(
        """
        SELECT entity_id
        FROM exact_index
        WHERE source = ?
          AND field = ?
          AND country = ?
          AND value = ?
        """,
        (
            source,
            field,
            country,
            value,
        ),
    ).fetchall()

    return {
        row[0]
        for row in rows
    }


def get_token_candidates(
    connection,
    source,
    country,
    field,
    value,
):
    candidates = set()

    for token in tokenize(value):

        rows = connection.execute(
            """
            SELECT entity_id
            FROM token_index
            WHERE source = ?
              AND field = ?
              AND country = ?
              AND token = ?
            """,
            (
                source,
                field,
                country,
                token,
            ),
        ).fetchall()

        for row in rows:
            candidates.add(row[0])

    return candidates


def generate_candidates_for_entity(
    connection,
    source1_row,
):
    candidates = set()

    country = str(
        source1_row.get(
            "country",
            "",
        )
    ).strip().lower()

    name = normalize_blocking_key(
        source1_row.get(
            "business_name",
            "",
        )
    )

    address = normalize_blocking_key(
        source1_row.get(
            "business_address",
            "",
        )
    )

    for source in ("S2", "S3"):

        candidates.update(
            get_exact_candidates(
                connection,
                source,
                country,
                "name",
                name,
            )
        )

        candidates.update(
            get_exact_candidates(
                connection,
                source,
                country,
                "address",
                address,
            )
        )

        candidates.update(
            get_token_candidates(
                connection,
                source,
                country,
                "name",
                name,
            )
        )

        candidates.update(
            get_token_candidates(
                connection,
                source,
                country,
                "address",
                address,
            )
        )

    return candidates


def generate_candidate_pairs(
    source1_df,
):
    connection = sqlite3.connect(
        DB_PATH
    )

    try:

        candidate_pairs = {}

        for _, row in source1_df.iterrows():

            source1_id = row["entity_id"]

            candidates = generate_candidates_for_entity(
                connection,
                row,
            )

            candidate_pairs[
                source1_id
            ] = candidates

        return candidate_pairs

    finally:

        connection.close()