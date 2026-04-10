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


# Ændringer:
# - filter_structured_data() beholder nu rækker med kreditkort,
#   selvom teksten er semi-struktureret (fx Nemotrons markdown-tabeller)
# - Ny augment_credit_cards(): duplikerer kreditkort-rækker med
#   modsatte format (med/uden mellemrum) så modellen lærer begge

import re
import ast
import pandas as pd
from datasets import load_dataset

# ── Labels vi er trænet til at genkende ──────────────────────────────────────
LABEL_MAP = {
    "O", 
    "API_KEY", 
    "CREDIT_CARD_NUMBER", 
    "BANK_ACCOUNT_NUMBER",
    "IBAN", 
    "PASSWORD", 
    "SSN",
    "FULL_NAME", 
    "FIRST_NAME", 
    "LAST_NAME", 
    "EMAIL", 
    "PHONE_NUMBER",
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
    "CREDIT_DEBIT_CARD":      "CREDIT_CARD_NUMBER",
    "IBAN":                   "IBAN",
    "PASSWORD":               "PASSWORD",
    "API_KEY":                "API_KEY",
    "BANK_ACCOUNT":           "BANK_ACCOUNT_NUMBER",
    "BANK_ACCOUNT_NUMBER":    "BANK_ACCOUNT_NUMBER",
}


# ── Filter: fjern struktureret data (JSON/CSV/logs) ──────────────────────────

def _has_pii_label(entities: list, label: str) -> bool:
    """Tjek om en liste af entiteter indeholder et bestemt label."""
    return any(e.get("label") == label for e in entities)


def filter_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fjerner tekster der ligner struktureret data (JSON, CSV, logs),
    MEN beholder rækker der indeholder kreditkort, da Nemotrons
    kreditkort-eksempler ofte er i semi-struktureret format.
    """
    regex_pattern = re.compile(r'\{.*\}|\[.*\]|".*"\s*:', re.DOTALL)
    special_chars = re.compile(r'[\{\}\[\]":;\|]')

    def is_structured(text: str) -> bool:
        if regex_pattern.search(text):
            return True
        density = len(special_chars.findall(text)) / max(len(text), 1)
        return density > 0.05

    structured_mask = df["source_text"].apply(is_structured)
    has_cc_mask     = df["privacy"].apply(lambda e: _has_pii_label(e, "CREDIT_CARD_NUMBER"))

    # Fjern struktureret data, MEN behold rækker med kreditkort
    remove_mask = structured_mask & ~has_cc_mask

    n_removed    = remove_mask.sum()
    n_kept_cc    = (structured_mask & has_cc_mask).sum()
    n_total      = len(df)

    print(f"Struktureret data fjernet: {n_removed:,} rækker "
          f"({n_removed / n_total * 100:.1f}% af datasættet)")
    print(f"  Beholdt {n_kept_cc:,} strukturerede rækker med kreditkort")

    return df[~remove_mask].copy()


# ── Kreditkort-normalisering ─────────────────────────────────────────────────

def _normalize_cc(number: str) -> str:
    """Fjerner mellemrum og bindestreger fra et kreditkortnummer."""
    return re.sub(r'[\s\-]', '', number)


def _add_cc_spaces(number: str) -> str:
    """Tilføjer mellemrum i grupper af 4 til et kreditkortnummer."""
    clean = _normalize_cc(number)
    return ' '.join(clean[i:i+4] for i in range(0, len(clean), 4))


def augment_credit_cards(df: pd.DataFrame) -> pd.DataFrame:
    """
    For hver række med kreditkort, opret en ekstra kopi hvor
    kreditkortnummeret har det modsatte format:
      - Hvis originalen har mellemrum → kopi uden mellemrum
      - Hvis originalen ikke har mellemrum → kopi med mellemrum

    Dette sikrer at modellen ser begge formater under træning.
    """
    new_rows = []

    for _, row in df.iterrows():
        entities = row["privacy"]
        if not _has_pii_label(entities, "CREDIT_CARD_NUMBER"):
            continue

        text         = row["source_text"]
        new_text     = text
        new_entities = []

        for entity in entities:
            if entity.get("label") == "CREDIT_CARD_NUMBER":
                original = entity["value"]
                clean    = _normalize_cc(original)

                # Bestem modsatte format
                if ' ' in original or '-' in original:
                    # Original har separatorer → lav version uden
                    alternate = clean
                else:
                    # Original har ingen separatorer → lav version med mellemrum
                    alternate = _add_cc_spaces(clean)

                new_text = new_text.replace(original, alternate, 1)
                new_entities.append({"label": "CREDIT_CARD_NUMBER", "value": alternate})
            else:
                new_entities.append(entity.copy())

        # Kun tilføj hvis teksten faktisk ændrede sig
        if new_text != text:
            new_rows.append({
                "source_text": new_text,
                "privacy":     new_entities,
            })

    if new_rows:
        aug_df = pd.DataFrame(new_rows, columns=["source_text", "privacy"])
        print(f"Kreditkort-augmentering: {len(aug_df):,} ekstra rækker tilføjet")
        return pd.concat([df, aug_df], ignore_index=True)

    return df


# ── Indlæsning: syvai/pii-dataset-eng ────────────────────────────────────────

def load_syvai() -> pd.DataFrame:
    """
    Indlæser alle rækker fra syvai/pii-dataset-eng fra Hugging Face.
    """
    print("Indlæser syvai/pii-dataset-eng...")
    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(file_path)

    df = df[["source_text", "privacy"]].dropna(subset=["source_text"])

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
    Indlæser alle rækker fra begge datasæt, kombinerer dem,
    fjerner struktureret data (men beholder kreditkort-rækker),
    og augmenterer kreditkortnumre med begge formater.
    """
    df_syvai    = load_syvai()
    df_nemotron = load_nemotron()

    df = pd.concat([df_syvai, df_nemotron], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nKombineret datasæt: {len(df):,} rækker i alt (blandet)")

    df = filter_structured_data(df)
    print(f"Datasæt efter filtrering: {len(df):,} rækker")

    df = augment_credit_cards(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Endeligt datasæt: {len(df):,} rækker\n")

    return df


# ── Hurtig test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_combined_dataset()
    print(df.head())
    print(f"\nKolonner: {df.columns.tolist()}")
    print(f"Eksempel privacy: {df['privacy'].iloc[0]}")