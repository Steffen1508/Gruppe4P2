"""
Analyse af syvai/pii-dataset-eng fra HuggingFace
- Tæller forekomster af hvert label
- Finder rækker med manglende data
"""

from datasets import load_dataset
from collections import Counter
import pandas as pd
import numpy as np

# ── Indlæs datasæt ────────────────────────────────────────────────────────────
print("Henter datasæt fra HuggingFace...")
ds = load_dataset("syvai/pii-dataset-eng")
print(f"Splits fundet: {list(ds.keys())}\n")

for split_name, split_data in ds.items():
    print("=" * 60)
    print(f"  SPLIT: {split_name}  ({len(split_data):,} rækker)")
    print("=" * 60)

    df = split_data.to_pandas()
    print(f"\nKolonner: {list(df.columns)}\n")

    # ── Manglende data ────────────────────────────────────────────────────────
    print("── Manglende data (None / NaN / tom streng) ──")
    missing_report = {}
    for col in df.columns:
        n_null = df[col].isna().sum()
        n_empty = 0
        if df[col].dtype == object:
            n_empty = (df[col].astype(str).str.strip() == "").sum() - n_null
        missing_report[col] = {"null/NaN": int(n_null), "tom streng": int(n_empty)}

    missing_df = pd.DataFrame(missing_report).T
    missing_df["total_mangler"] = missing_df["null/NaN"] + missing_df["tom streng"]
    print(missing_df.to_string())

    rows_with_any_missing = df[df.isna().any(axis=1)]
    print(f"\nAntal rækker med mindst ét manglende felt: {len(rows_with_any_missing):,}")

    # ── Label-fordeling ───────────────────────────────────────────────────────
    print("\n── Label-fordeling ──")

    label_col = "privacy"
    all_labels = []
    rows_without_labels = 0

    for row in df[label_col]:
        # Håndter både Python list, numpy array og andre iterables
        try:
            entities = list(row) if not isinstance(row, float) else []
        except TypeError:
            entities = []

        if len(entities) == 0:
            rows_without_labels += 1
        for entity in entities:
            # entity kan være dict eller numpy void/structured array
            if isinstance(entity, dict):
                lbl = entity.get("label")
            else:
                try:
                    lbl = entity["label"]
                except (KeyError, IndexError, TypeError):
                    lbl = None
            if lbl:
                all_labels.append(lbl)

    print(f"Rækker uden PII-labels (ingen entities): {rows_without_labels:,}\n")

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())

    print(f"{'Label':<35} {'Antal':>10} {'%':>8}")
    print("-" * 57)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"{label:<35} {count:>10,} {pct:>7.2f}%")
    print("-" * 57)
    print(f"{'TOTAL':<35} {total:>10,} {'100.00%':>8}")

    print()