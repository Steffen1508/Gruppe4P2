import pandas as pd

def main():
    try:
        # Indlæs dataset
        file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"
        df = pd.read_parquet(file_path)

        print("\n===== DATA OVERVIEW =====")
        print(df.head(10))
        print("\n===== DIMENSIONER =====")
        print(f"Antal rækker: {df.shape[0]}")
        print(f"Antal kolonner: {df.shape[1]}")

        print("\n===== DATATYPER =====")
        print(df.dtypes)

        print("\n===== MISSING VALUES =====")
        print(df.isnull().sum())

        print("\n===== DATA INFO =====")
        df.info()
        
        print("\n===== TEKST LÆNGDE ANALYSE =====")
        df["text_length"] = df["source_text"].str.len()
        print(df["text_length"].describe())

        print("\n===== ANTAL ORD =====")
        df["word_count"] = df["source_text"].str.split().str.len() 
        print(df["word_count"].describe()) 
         
    
        print("\n===== ANTAL PRIVACY LABELS PR RÆKKE =====")
        df["num_privacy_tags"] = df["privacy"].apply(len)
        print(df["num_privacy_tags"].describe())  # Resultatet er antal PII-labels pr. tekst.

    except Exception as e:
        print("Der opstod en fejl:", e)

if __name__ == "__main__":
    main()
    