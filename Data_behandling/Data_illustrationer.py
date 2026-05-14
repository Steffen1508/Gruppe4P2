import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sæt et pænt tema for alle grafer (ser godt ud i LaTeX-rapporter)
sns.set_theme(style="whitegrid")

def plot_label_distribution(df_flat):
    """
    Figur 1: Viser fordelingen af de 13 valgte PII-labels.
    Argument i rapporten: Viser hvilke klasser der har meget data (f.eks. EMAIL), 
    og hvilke der er svære for modellen at lære pga. manglende data (f.eks. PASSPORT_NUMBER).
    """
    plt.figure(figsize=(12, 6))
    
    # Tæl forekomsten af hvert label (Eksklusiv "O")
    label_counts = df_flat['label'].value_counts()
    
    # Byg et Bar Chart
    ax = sns.barplot(x=label_counts.values, y=label_counts.index, hue=label_counts.index, palette="viridis", legend=False)
    
    plt.title("Fordeling af PII-Entiteter i det Kombinerede Datasæt", fontsize=16)
    plt.xlabel("Antal Forekomster", fontsize=12)
    plt.ylabel("PII Kategori", fontsize=12)
    
    # Gem figuren i høj opløsning (dpi=300 er standard for akademiske rapporter)
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=300)
    print("Gemte 'label_distribution.png'")
    plt.close()

def plot_class_imbalance(df, df_flat):
    """
    Figur 2: Viser PII-ord vs. 'Outside' (O) ord.
    Argument i rapporten: Det visuelle bevis for The Semantic Gap og 
    nødvendigheden af at bruge Class Weights i BERT for at undgå over-predicting af 'O'.
    """
    plt.figure(figsize=(8, 8))
    
    # Estimer antal ord
    total_words = df['source_text'].str.split().str.len().sum()
    
    # Tæl PII ord (antal ord i alle 'value' felter i privacy-kolonnen)
    df_flat['pii_word_count'] = df_flat['value'].astype(str).str.split().str.len()
    total_pii_words = df_flat['pii_word_count'].sum()
    
    outside_words = total_words - total_pii_words
    
    # Data til lagkagediagram (Pie Chart)
    labels = ['Outside Tokens (O)', 'PII Tokens']
    sizes = [outside_words, total_pii_words]
    colors = ['#d3d3d3', '#ff9999']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
            explode=(0, 0.1), textprops={'fontsize': 14})
    
    plt.title("Klasse-ubalance: Non-PII vs. PII Tokens", fontsize=16)
    
    plt.tight_layout()
    plt.savefig("class_imbalance.png", dpi=300)
    print("Gemte 'class_imbalance.png'")
    plt.close()

def plot_document_lengths(df):
    """
    Figur 3: Viser fordelingen af dokumentlængder (antal ord).
    Argument i rapporten: Beviser hvorfor 'max_len = 128' er valgt i BERT, 
    og illustrerer behovet for Chunking af ekstremt lange tekster.
    """
    plt.figure(figsize=(10, 5))
    
    # Beregn antal ord per række
    df['word_count'] = df['source_text'].str.split().str.len()
    
    # Filtrer ekstreme outliers fra grafen for at gøre den pænere (f.eks. over 500 ord)
    plot_data = df[df['word_count'] <= 500]
    
    sns.histplot(plot_data['word_count'], bins=50, kde=True, color="teal")
    
    # Indsæt en rød stiplet linje der viser jeres BERT max_len
    plt.axvline(x=128, color='red', linestyle='--', label='BERT max_len (128 tokens)')
    
    plt.title("Fordeling af Dokumentlængder i Træningsdata", fontsize=16)
    plt.xlabel("Antal Ord", fontsize=12)
    plt.ylabel("Antal Dokumenter", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("document_lengths.png", dpi=300)
    print("Gemte 'document_lengths.png'")
    plt.close()

from data_loader import load_combined_dataset

def main():
    print("Indlæser det faste, lokalt gemte datasæt...")
    # Loader fra pre-defineret data loader
    df = load_combined_dataset()
    
    # For label distribution har vi brug for at 'explode' privacy kolonnen
    # så vi har én række pr. label, i stedet for en liste af dicts pr. tekst
    print("Udpakker JSON-struktur til plot...")
    df_flat = df.explode('privacy').dropna(subset=['privacy']).reset_index(drop=True)
    pii_details = pd.json_normalize(df_flat['privacy'])
    
    # Kør funktionerne
    print("Genererer figurer...")
    plot_label_distribution(pii_details)
    plot_class_imbalance(df, pii_details)
    plot_document_lengths(df)
    
    print("Færdig! Du kan nu kopiere .png filerne ind i LaTeX.")

if __name__ == "__main__":
    main()