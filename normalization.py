import re
import unicodedata


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # Replace punctuation with spaces while preserving
    # Unicode letters and numbers.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = value.replace("_", " ")

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


def normalize_country(value):
    return normalize_text(value)


def normalize_entity_dataframe(df):
    """
    Add normalized versions of name, address and country.
    """

    result = df.copy()

    result["name_normalized"] = (
        result["business_name"]
        .map(normalize_name)
    )

    result["address_normalized"] = (
        result["business_address"]
        .map(normalize_address)
    )

    result["country_normalized"] = (
        result["country"]
        .map(normalize_country)
    )

    return result