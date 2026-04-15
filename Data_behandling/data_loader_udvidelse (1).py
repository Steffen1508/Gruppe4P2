# Make sure 'data_loader.py' is in the same directory before running!
# Requirements: pip install datasets pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_combined_dataset

def main():
    # ==========================================
    # 1. LOAD DATA (In-Memory)
    # ==========================================
    print("Loading and mixing Syvai + Nemotron datasets (this may take a moment)...")
    df = load_combined_dataset()

    # ==========================================
    # 2. DATA ENGINEERING & TRANSFORMATION
    # ==========================================
    print("Performing data engineering...")

    # Store metadata before transformation
    df['tags_per_msg'] = df['privacy'].apply(len)
    df['msg_length'] = df['source_text'].str.len()

    # Transformation: Explode nested JSON lists into separate rows
    df_flat = df.explode('privacy').reset_index(drop=True)
    df_flat = df_flat.dropna(subset=['privacy']).reset_index(drop=True)
    pii_details = pd.json_normalize(df_flat['privacy'])

    # Assemble final dataframe
    df_final = pd.concat([
        df_flat[['source_text', 'tags_per_msg', 'msg_length']],
        pii_details
    ], axis=1)

    # Sanitization: Clean PII values
    df_final['value'] = df_final['value'].astype(str).str.replace('|', '', regex=False).str.strip()
    df_final = df_final.dropna(subset=['label', 'value'])
    df_final = df_final[df_final['value'] != ""]

    # PII value lengths (used for optional analysis)
    df_final['value_len'] = df_final['value'].str.len()

    # ==========================================
    # 3. GENERATE FIGURES FOR THE REPORT
    # ==========================================
    print("Generating Figure 3.2 and 3.3 for the report...")
    sns.set_theme(style="whitegrid")

    # ==========================================
    # --- FIGURE 3.2: Major Classes (>= 1000) ---
    # ==========================================
    print("Generating label distributions...")

    # Count all observations per label
    label_counts = df_final['label'].value_counts()

    # Split into two groups based on threshold (1000)
    major_labels = label_counts[label_counts >= 1000]
    minor_labels = label_counts[label_counts < 1000]

    if not major_labels.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(y=major_labels.values, x=major_labels.index, palette='viridis')
        plt.title('Figure 3.2: PII Label Distribution (Major Classes >= 1000)')
        plt.ylabel('Count')
        plt.xlabel('PII Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figure_3_2_label_distribution_major.png', dpi=300)
        plt.close()
        print("  -> figure_3_2_label_distribution_major.png saved")
    else:
        print("  No labels have >= 1000 observations.")

    # ==========================================
    # --- FIGURE 3.3: Minor Classes (< 1000) ---
    # ==========================================
    if not minor_labels.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(y=minor_labels.values, x=minor_labels.index, palette='magma')
        plt.title('Figure 3.3: PII Label Distribution (Minor Classes < 1000)')
        plt.ylabel('Count')
        plt.xlabel('PII Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figure_3_3_label_distribution_minor.png', dpi=300)
        plt.close()
        print("  -> figure_3_3_label_distribution_minor.png saved")
    else:
        print("  No labels have fewer than 1000 observations.")

    # ==========================================
    # 4. SUMMARY
    # ==========================================
    print(f"\nSummary:")
    print(f"  Total rows in dataset:        {len(df):,}")
    print(f"  Total PII entities (flat):    {len(df_final):,}")
    print(f"  Unique PII labels:            {df_final['label'].nunique()}")
    print(f"  Labels: {sorted(df_final['label'].unique())}")

    print("\nDone! The following images are now saved in the working directory:")
    print("  - figure_3_2_label_distribution_major.png")
    print("  - figure_3_3_label_distribution_minor.png")

if __name__ == "__main__":
    main()