import pandas as pd
import re

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

def analyze_and_clean_text_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyserer og filtrerer tekster, der ligner struktureret data (CSV/JSON/Logs).
    Printer eksempler på de fjernede observationer og returnerer det rensede datasæt.
    """
    print("\n==========================================")
    print("--- 1.5 Analyse og Rensning af Tekststruktur (Anomalier) ---")
    
    # Metode 1: Regex til tydelig JSON/Array struktur
    regex_mask = df['source_text'].str.contains(r'\{.*\}|\[.*\]|".*"\s*:', regex=True, na=False)
    
    # Metode 2: Character Density (Tegndensitet)
    def get_special_char_density(text):
        text = str(text)
        if len(text) == 0: return 0
        special_chars = len(re.findall(r'[\{\}\[\]":;\|]', text))
        return special_chars / len(text)

    # Filtrer tekster hvor mere end 5% af indholdet er kode-tegn
    density_mask = df['source_text'].apply(get_special_char_density) > 0.05
    
    # Kombiner maskerne (behold rækker, der IKKE matcher nogen af anomali-filtrene)
    combined_mask = regex_mask | density_mask
    
    df_structured = df[combined_mask]
    df_natural = df[~combined_mask].copy() # .copy() forhindrer SettingWithCopyWarning senere
    
    structured_count = len(df_structured)
    print(f"Antal tekster identificeret som struktureret data: {structured_count:,.0f}")
    print(f"Procentdel af hele datasættet: {(structured_count / len(df)) * 100:.2f}%\n")
    
    # Udskriv eksempler på det, vi smider væk
    if structured_count > 0:
        print("--- EKSEMPLER PÅ FJERNET DATA (Top 3) ---")
        for i, text in enumerate(df_structured['source_text'].head(3)):
            print(f"\n[Fjernet Eksempel {i+1}]:")
            preview = text[:250].replace('\n', ' ')
            print(f"{preview}..." if len(text) > 250 else preview)
        print("\n-----------------------------------------")
        
    print("==========================================\n")
    return df_natural

def analyze_transformed_data(df: pd.DataFrame) -> None:
    """Udfører analyse af dataformatet lige efter nested JSON er fladet ud."""
    print("\n==========================================")
    print("--- 2. Analyse af det TRANSFORMEREDE Datasæt (EFTER transformation) ---")
    print(f"Nyt antal rækker (hver række er nu et specifikt PII-tag): {df.shape[0]:,.0f}")
    print(f"Nyt antal kolonner: {df.shape[1]}")
    print("\n--- De nye flade kolonner ---")
    df.info()
    print("==========================================\n")

def calculate_true_imbalance(df_flat: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Estimerer forholdet mellem PII-tokens og 'Outside' (O) tokens."""
    print("\n==========================================")
    print("--- 3. Den Sande Klasse-ubalance (PII vs. Non-PII) ---")
    total_words = df_flat['source_text'].str.split().str.len().sum()
    df_clean['pii_word_count'] = df_clean['value'].str.split().str.len()
    total_pii_words = df_clean['pii_word_count'].sum()
    outside_words = total_words - total_pii_words
    
    print(f"Totalt antal ord (estimat): {total_words:,.0f}")
    print(f"Antal PII-ord i valgte domæner: {total_pii_words:,.0f} ({(total_pii_words/total_words)*100:.2f}%)")
    print(f"Antal 'Outside' (O) ord: {outside_words:,.0f} ({(outside_words/total_words)*100:.2f}%)")
    print("==========================================\n")


# ==========================================
# 2. HOVEDSCRIPT (Data Pipeline)
# ==========================================

if __name__ == "__main__":
    # --- A. Indlæsning ---
    print("Henter rådata...")
    df_raw = pd.read_parquet("hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet")

    # --- B. Analyse FØR transformation ---
    analyze_raw_data(df_raw)

    # --- NYT TRIN: Rensning for struktureret data (FØR udpakning) ---
    print("Kører struktur-analyse og filtrerer data...")
    df = analyze_and_clean_text_structure(df_raw)

    # Gem metadata før vi ændrer antallet af rækker (nu på det naturlige datasæt)
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

    # --- G. Kør ubalance analyse på det nu rensede dataset ---
    calculate_true_imbalance(df_flat, df_clean)
    
    print("\n--- Script færdigkørt ---")