import pandas as pd
from typing import Tuple, Dict


DATA_PATH = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"
TARGET_LABELS = {"FULL_NAME", "EMAIL", "PHONE_NUMBER", "STREET_ADDRESS", "CITY"}


def print_title(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_dataset(path: str) -> pd.DataFrame:
    print("Indlæser datasæt...")
    return pd.read_parquet(path)


def report_dataframe_state(df: pd.DataFrame, title: str) -> None:
    print_title(title)
    print(f"Antal rækker: {len(df):,}")
    print(f"Antal kolonner: {df.shape[1]}")
    print("\nKolonneoversigt og datatyper:")
    df.info()
    print()


def add_message_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["tags_per_message"] = enriched["privacy"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    enriched["char_length"] = enriched["source_text"].astype(str).str.len()
    enriched["word_length"] = enriched["source_text"].astype(str).str.split().str.len()
    return enriched


def flatten_annotations(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.explode("privacy", ignore_index=True)
    annotation_cols = pd.json_normalize(exploded["privacy"])
    flat = pd.concat(
        [
            exploded[["source_text", "tags_per_message", "char_length", "word_length"]],
            annotation_cols,
        ],
        axis=1,
    )
    return flat


def clean_flattened_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned["value"] = (
        cleaned["value"]
        .astype(str)
        .str.replace("|", "", regex=False)
        .str.strip()
    )

    cleaned = cleaned.dropna(subset=["label", "value"])
    cleaned = cleaned[(cleaned["label"] != "") & (cleaned["value"] != "")]
    return cleaned


def show_top_labels(df: pd.DataFrame, n: int = 10) -> None:
    print_title(f"Top {n} mest hyppige PII labels")
    print(df["label"].value_counts().head(n))


def analyze_long_texts(df: pd.DataFrame, threshold_words: int = 400) -> None:
    long_mask = df["word_length"] > threshold_words
    long_count = int(long_mask.sum())
    total = len(df)

    print_title("Lange tekster og potentiel chunking")
    print(f"Threshold: > {threshold_words} ord")
    print(f"Antal rækker over threshold: {long_count:,}")
    print(f"Andel af datasættet: {(long_count / total) * 100:.2f}%")


def filter_domain_labels(df: pd.DataFrame, labels: set[str]) -> pd.DataFrame:
    filtered = df[df["label"].isin(labels)].copy()
    filtered["entity_char_length"] = filtered["value"].str.len()
    filtered["entity_word_length"] = filtered["value"].str.split().str.len()
    return filtered


def inspect_structured_text(df: pd.DataFrame) -> None:
    print_title("Strukturel analyse af source_text")
    pattern = r"\{.*\}|\[.*\]|\".*\"\s*,"

    structured_like = df["source_text"].astype(str).str.contains(
        pattern,
        regex=True,
        na=False
    )
    count = int(structured_like.sum())

    print(f"Antal tekster der ligner struktureret data: {count:,}")
    print(f"Andel af datasættet: {(count / len(df)) * 100:.2f}%")


def estimate_token_imbalance(
    full_flat_df: pd.DataFrame,
    filtered_df: pd.DataFrame
) -> None:
    print_title("Estimat af klasseubalance mellem PII og O")

    total_words = full_flat_df["source_text"].astype(str).str.split().str.len().sum()

    pii_word_total = (
        filtered_df["value"]
        .astype(str)
        .str.split()
        .str.len()
        .sum()
    )

    outside_words = total_words - pii_word_total

    pii_pct = (pii_word_total / total_words) * 100 if total_words else 0
    outside_pct = (outside_words / total_words) * 100 if total_words else 0

    print(f"Samlet antal ord i source_text: {total_words:,}")
    print(f"PII ord i valgte labels: {pii_word_total:,} ({pii_pct:.2f}%)")
    print(f"O ord estimeret: {outside_words:,} ({outside_pct:.2f}%)")


def dataset_summary(raw_df: pd.DataFrame, flat_df: pd.DataFrame, domain_df: pd.DataFrame) -> None:
    print_title("Kort opsummering")
    print(f"Rå dokumenter: {len(raw_df):,}")
    print(f"Fladgjorte annotationer: {len(flat_df):,}")
    print(f"Rækker i Combine CDC delmængde: {len(domain_df):,}")
    print(f"Antal unikke labels i hele datasættet: {flat_df['label'].nunique():,}")


def main() -> None:
    raw_df = load_dataset(DATA_PATH)

    report_dataframe_state(
        raw_df,
        "1. Analyse af råt datasæt før feature engineering"
    )

    enriched_df = add_message_features(raw_df)

    print("Flader annotationsstrukturen ud...")
    flat_df = flatten_annotations(enriched_df)

    report_dataframe_state(
        flat_df,
        "2. Analyse efter flattening af annotationsdata"
    )

    flat_df = clean_flattened_data(flat_df)

    show_top_labels(flat_df, n=10)
    analyze_long_texts(flat_df, threshold_words=400)

    domain_df = filter_domain_labels(flat_df, TARGET_LABELS)

    print_title("Filtrering til Combine CDC domæne")
    print(f"Antal rækker efter domænefiltrering: {len(domain_df):,}")

    inspect_structured_text(raw_df)
    estimate_token_imbalance(flat_df, domain_df)
    dataset_summary(raw_df, flat_df, domain_df)

    print("\nAnalyse afsluttet.")


if __name__ == "__main__":
    main()