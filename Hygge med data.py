import pandas as pd

# ==========================================
# 1. FUNKTIONSDEFINITIONER (Dine analyseværktøjer)
# ==========================================

def analyze_raw_data(df: pd.DataFrame) -> None:
    """Udfører en grundlæggende EDA af det helt rå datasæt før transformation."""
    print("\n==========================================")
    print("--- 1. Analyse af det RÅ Datasæt (FØR transformation) ---")
    print(f"Oprindeligt antal rækker (beskeder): {df.shape[0]:,.0f}")
    print(f"Oprindeligt antal kolonner: {df.shape[1]}")
    print("\n--- Datatyper og Non-Null værdier ---")
    df.info() 
    print("==========================================\n")

def analyze_transformed_data(df: pd.DataFrame) -> None:
    """Udfører analyse af dataformatet lige efter nested JSON er fladet ud."""
    print("\n==========================================")
    print("--- 2. Analyse af det TRANSFORMEREDE Datasæt (EFTER transformation) ---")
    print(f"Nyt antal rækker (hver række er nu et specifikt PII-tag): {df.shape[0]:,.0f}")
    print(f"Nyt antal kolonner: {df.shape[1]}")
    print("\n--- De nye flade kolonner ---")
    df.info()
    print("==========================================\n")

def analyze_text_structure(df: pd.DataFrame) -> None:
    """Analyserer om teksterne ligner fritekst eller struktureret data (CSV/JSON)."""
    print("\n--- 3. Analyse af tekststruktur (Anomalier) ---")
    structured_mask = df['source_text'].str.contains(r'\{.*\}|\[.*\]|".*"\s*,', regex=True, na=False)
    structured_count = structured_mask.sum()
    print(f"Antal tekster der ligner struktureret data (CSV/JSON): {structured_count:,.0f}")
    print(f"Procentdel af hele datasættet: {(structured_count / len(df)) * 100:.2f}%\n")

def calculate_true_imbalance(df_flat: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Estimerer forholdet mellem PII-tokens og 'Outside' (O) tokens."""
    print("\n--- 4. Den Sande Klasse-ubalance (PII vs. Non-PII) ---")
    total_words = df_flat['source_text'].str.split().str.len().sum()
    df_clean['pii_word_count'] = df_clean['value'].str.split().str.len()
    total_pii_words = df_clean['pii_word_count'].sum()
    outside_words = total_words - total_pii_words
    
    print(f"Totalt antal ord (estimat): {total_words:,.0f}")
    print(f"Antal PII-ord i valgte domæner: {total_pii_words:,.0f} ({(total_pii_words/total_words)*100:.2f}%)")
    print(f"Antal 'Outside' (O) ord: {outside_words:,.0f} ({(outside_words/total_words)*100:.2f}%)")


# ==========================================
# 2. HOVEDSCRIPT (Data Pipeline)
# ==========================================

if __name__ == "__main__":
    # --- A. Indlæsning ---
    print("Henter rådata...")
    df = pd.read_parquet("hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet")

    # --- B. Analyse FØR transformation ---
    analyze_raw_data(df)

    # Gem metadata før vi ændrer antallet af rækker
    df['tags_per_msg'] = df['privacy'].apply(len)
    df['msg_length'] = df['source_text'].str.len()

    # --- C. Transformation (Flattening nested JSON) ---
    print("Transformerer data via explode()...")
    df_flat = df.explode('privacy').reset_index(drop=True)
    pii_details = pd.json_normalize(df_flat['privacy'])

    df_final = pd.concat([
        df_flat[['source_text', 'tags_per_msg', 'msg_length']], 
        pii_details
    ], axis=1)

    # --- D. Analyse EFTER transformation ---
    analyze_transformed_data(df_final)

    print("\nTop 10 hyppigste PII-typer i det transformerede datasæt:")
    print(df_final['label'].value_counts().head(10))

    # --- E. Grundlæggende rensning ---
    df_final['value'] = df_final['value'].astype(str).str.replace('|', '', regex=False).str.strip()
    df_final = df_final.dropna(subset=['label', 'value'])
    df_final = df_final[df_final['value'] != ""]

    df_final['word_count'] = df_final['source_text'].str.split().str.len()
    long_texts = df_final[df_final['word_count'] > 400]
    print(f"\nAntal rækker der kræver chunking (>400 ord): {len(long_texts)} ({len(long_texts)/len(df_final)*100:.2f}%)")

    # --- F. Filtrering til jeres specifikke Combine CDC case ---
    relevant_labels = ['FULL_NAME', 'EMAIL', 'PHONE_NUMBER', 'STREET_ADDRESS', 'CITY']
    df_clean = df_final[df_final['label'].isin(relevant_labels)].copy()
    
    # Beregn længden på værdierne i det filtrerede dataset
    df_clean['value_len'] = df_clean['value'].str.len()

    print(f"\nAntal rækker efter filtrering til Combine CDC domæne: {len(df_clean)}")

    # --- G. Kør de avancerede EDA funktioner ---
    analyze_text_structure(df_flat)
    calculate_true_imbalance(df_flat, df_clean)
    
    print("\n--- Script færdigkørt ---")