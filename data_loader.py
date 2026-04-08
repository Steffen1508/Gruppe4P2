# data_loader.py
# ─────────────────────────────────────────────
# Indlæser og kombinerer to PII-datasæt fra Hugging Face:
#   - syvai/pii-dataset-eng
#   - nvidia/Nemotron-PII
#
# Returnerer én samlet DataFrame med kolonnerne:
#   source_text : str              – den rå tekst
#   privacy     : list[dict]       – [{"label": ..., "value": ...}, ...]
#
# Brug:
#   from data_loader import load_combined_dataset
#   df = load_combined_dataset()
#
# Kræver: pip install datasets pandas
# ─────────────────────────────────────────────

import re
import ast
import pandas as pd
from datasets import load_dataset

# ── Labels vi er trænet til at genkende ──────────────────────────────────────
LABEL_MAP = {
    "O", "API_KEY", "CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER",
    "IBAN", "PASSWORD", "PASSPORT_NUMBER", "SSN",
    "FULL_NAME", "FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE_NUMBER",
}

# ── Mapping fra Nemotron-labels til vores kategorier ─────────────────────────
NEMOTRON_LABEL_MAP = {
    "NAME":                   "FULL_NAME",
    "FULL_NAME":              "FULL_NAME",
    "FIRST_NAME":             "FIRST_NAME",
    "LAST_NAME":              "LAST_NAME",
    "EMAIL":                  "EMAIL",
    "EMAIL_ADDRESS":          "EMAIL",
    "PHONE":                  "PHONE_NUMBER",
    "PHONE_NUMBER":           "PHONE_NUMBER",
    "SSN":                    "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "CREDIT_CARD":            "CREDIT_CARD_NUMBER",
    "CREDIT_CARD_NUMBER":     "CREDIT_CARD_NUMBER",
    "IBAN":                   "IBAN",
    "PASSWORD":               "PASSWORD",
    "API_KEY":                "API_KEY",
    "PASSPORT":               "PASSPORT_NUMBER",
    "PASSPORT_NUMBER":        "PASSPORT_NUMBER",
    "BANK_ACCOUNT":           "BANK_ACCOUNT_NUMBER",
    "BANK_ACCOUNT_NUMBER":    "BANK_ACCOUNT_NUMBER",
    # Labels uden for vores model ignoreres
}


# ── Filter: fjern struktureret data (JSON/CSV/logs) ──────────────────────────

def filter_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fjerner tekster der ligner struktureret data (JSON, CSV, logs).
    Baseret på to kriterier:

    1. Regex-detektion: teksten indeholder JSON-objekter, arrays
       eller key-value par som {"key": "value"} eller [item].

    2. Tegndensitet: mere end 5% af tegnene er strukturelle
       specialtegn som {, }, [, ], :, ;, |, ".
       Naturlig tekst har typisk under 2% sådanne tegn.

    Disse tekster kan skade modellen fordi BERT risikerer at lære
    at genkende PII via strukturelle markører (kolon, anførselstegn)
    frem for sproglig kontekst – præcis det vi vil undgå.
    """
    regex_pattern = re.compile(r'\{.*\}|\[.*\]|".*"\s*:', re.DOTALL)
    special_chars = re.compile(r'[\{\}\[\]":;\|]')

    def is_structured(text: str) -> bool:
        if regex_pattern.search(text):
            return True
        density = len(special_chars.findall(text)) / max(len(text), 1)
        return density > 0.05

    mask      = df["source_text"].apply(is_structured)
    n_removed = mask.sum()
    n_total   = len(df)

    print(f"Struktureret data fjernet: {n_removed:,} rækker "
          f"({n_removed / n_total * 100:.1f}% af datasættet)")

    return df[~mask].copy()


# ── Indlæsning: syvai/pii-dataset-eng ────────────────────────────────────────

def load_syvai() -> pd.DataFrame:
    """
    Indlæser alle rækker fra syvai/pii-dataset-eng fra Hugging Face.

    Returnerer DataFrame med kolonnerne source_text og privacy.
    Kolonnen privacy indeholder lister på formen
        [{"label": "EMAIL", "value": "foo@bar.com"}, ...]
    som allerede er datasættets eget format.
    """
    print("Indlæser syvai/pii-dataset-eng...")
    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(file_path)

    df = df[["source_text", "privacy"]].dropna(subset=["source_text"])

    # Sikr at privacy altid er en liste
    df["privacy"] = df["privacy"].apply(
        lambda v: list(v) if v is not None else []
    )

    print(f"  Indlæste {len(df):,} rækker fra syvai")
    return df.reset_index(drop=True)


# ── Indlæsning: nvidia/Nemotron-PII ──────────────────────────────────────────

def _parse_nemotron_spans(raw) -> list:
    """
    Konverterer spans-kolonnen fra Nemotron til listen
        [{"label": ..., "value": ...}, ...]
    med kun de labels vores model kender.
    """
    if raw is None:
        return []

    try:
        spans = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except Exception:
        return []

    entities = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        raw_label = (
            span.get("label") or span.get("type") or span.get("entity_type") or ""
        )
        text = (
            span.get("text") or span.get("value") or span.get("span") or ""
        )
        if not raw_label or not text:
            continue

        mapped = NEMOTRON_LABEL_MAP.get(str(raw_label).upper())
        if mapped:
            entities.append({"label": mapped, "value": str(text)})

    return entities


def load_nemotron() -> pd.DataFrame:
    """
    Indlæser alle rækker fra nvidia/Nemotron-PII (alle locales).

    Returnerer DataFrame med kolonnerne source_text og privacy.
    """
    print("Indlæser nvidia/Nemotron-PII (train + test)...")
    rows = []

    for split in ("train", "test"):
        ds = load_dataset("nvidia/Nemotron-PII", split=split, streaming=True)
        split_count = 0

        for row in ds:
            text = row.get("text", "")
            if not text or not str(text).strip():
                continue

            rows.append({
                "source_text": str(text),
                "privacy":     _parse_nemotron_spans(row.get("spans")),
            })
            split_count += 1

        print(f"  {split}: {split_count:,} rækker")

    df = pd.DataFrame(rows, columns=["source_text", "privacy"])
    print(f"  Nemotron i alt: {len(df):,} rækker")
    return df


# ── Kombineret indlæsning ─────────────────────────────────────────────────────

def load_combined_dataset() -> pd.DataFrame:
    """
    Indlæser alle rækker fra begge datasæt, kombinerer dem og
    fjerner struktureret data (JSON/CSV/logs).

    Returns
    -------
    pd.DataFrame med kolonnerne source_text og privacy, klar til BertPipeline.
    """
    df_syvai    = load_syvai()
    df_nemotron = load_nemotron()

    df = pd.concat([df_syvai, df_nemotron], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nKombineret datasæt: {len(df):,} rækker i alt (blandet)")

    df = filter_structured_data(df)
    print(f"Datasæt efter filtrering: {len(df):,} rækker\n")

    return df


# ── Hurtig test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_combined_dataset()
    print(df.head())
    print(f"\nKolonner: {df.columns.tolist()}")
    print(f"Eksempel privacy: {df['privacy'].iloc[0]}")
