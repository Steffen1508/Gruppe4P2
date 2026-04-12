# FØR DU KØRER: Sørg for at 'data_loader.py' ligger i samme mappe!

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_combined_dataset

def main():
    # ==========================================
    # 1. HENT DATA DIREKTE (In-Memory)
    # ==========================================
    print("Henter og mixer Syvai + Nemotron datasættene (Dette tager lige et øjeblik)...")
    df = load_combined_dataset()

    # ==========================================
    # 2. DATA ENGINEERING & TRANSFORMATION
    # ==========================================
    print("Udfører data engineering...")
    
    # Gem metadata før transformation
    df['tags_per_msg'] = df['privacy'].apply(len)
    df['msg_length'] = df['source_text'].str.len()

    # Transformation: Udpak nested JSON-lister til separate rækker
    df_flat = df.explode('privacy').reset_index(drop=True)
    df_flat = df_flat.dropna(subset=['privacy']).reset_index(drop=True)
    pii_details = pd.json_normalize(df_flat['privacy'])

    # Samling af endelig dataframe
    df_final = pd.concat([
        df_flat[['source_text', 'tags_per_msg', 'msg_length']], 
        pii_details
    ], axis=1)

    # Sanitization: Rensning af PII-værdierne
    df_final['value'] = df_final['value'].astype(str).str.replace('|', '', regex=False).str.strip()
    df_final = df_final.dropna(subset=['label', 'value'])
    df_final = df_final[df_final['value'] != ""]

    # Vi kigger på PII-værdiernes længde (Bruges til Figur 3.3)
    df_final['value_len'] = df_final['value'].str.len()

    # ==========================================
    # 3. GENERER FIGURER TIL RAPPORTEN
    # ==========================================
    print("Genererer Figur 3.1, 3.2 og 3.3 til rapporten...")
    sns.set_theme(style="whitegrid")

    # ==========================================
    # --- FIGUR 3.1a og 3.1b: Opdelt Fordeling af PII-labels ---
    # ==========================================
    print("Genererer opdelte label-distributioner...")
    
    # 1. Tæl alle observationer per label
    label_counts = df_final['label'].value_counts()
    
    # 2. Split i to grupper baseret på jeres threshold (1000)
    major_labels = label_counts[label_counts >= 1000]
    minor_labels = label_counts[label_counts < 1000]

    # --- Figur 3.1a: Major Classes (>= 1000) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(y=major_labels.values, x=major_labels.index, palette='viridis')
    plt.title('Figur 3.1a: PII Label Distribution (Major Classes >= 1000)')
    plt.ylabel('Antal Forekomster')
    plt.xlabel('PII Kategori')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figur_3_1a_label_distribution_major.png', dpi=300)
    plt.close()

    # --- Figur 3.1b: Minor Classes (< 1000) ---
    # Vi gør grafen lidt lavere (figsize=(10, 4)), da der sandsynligvis er færre labels i denne gruppe
    if not minor_labels.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(y=minor_labels.values, x=minor_labels.index, palette='magma') # Skift farvepalette for at vise det er en ny gruppe
        plt.title('Figur 3.1b: PII Label Distribution (Minor Classes < 1000)')
        plt.ylabel('Antal Forekomster')
        plt.xlabel('PII Kategori')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figur_3_1b_label_distribution_minor.png', dpi=300)
        plt.close()
    else:
        print("Ingen labels har under 1000 observationer.")

    # --- FIGUR 3.2: Tekstlængde vs. Antal Tags (NFR1 Analysis) ---
    plt.figure(figsize=(10, 6))
    plt.hexbin(df_final['msg_length'], df_final['tags_per_msg'], gridsize=30, cmap='Blues', bins='log')
    plt.colorbar(label='log10(Antal rækker)')
    plt.title('Figur 3.2: Korrelation mellem Tekstlængde og Antal PII-tags')
    plt.xlabel('Beskedens længde (antal tegn)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Antal tags i beskeden')
    plt.tight_layout()
    plt.savefig('figur_3_2_length_vs_tags.png', dpi=300)
    plt.close()

  

    print("\nSucces! Følgende billeder ligger nu klar i mappen:")
    print("- figur_3_1_label_distribution.png")
    print("- figur_3_2_length_vs_tags.png")

if __name__ == "__main__":
    main()