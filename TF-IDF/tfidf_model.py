import os
import sys

# Gets the absolute path of this file, then navigates up two levels to find the project root folder.
# Adds the root directory to Python's path so we can import scripts (like data_loader.py) from outside this folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import itertools
import warnings

import joblib

from data_loader import load_combined_dataset

# Are the four lines below strictly necessary???
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Suppress non-critical Python warnings (like "failed to converge") that often spam the console during ML training.
warnings.filterwarnings("ignore")

# =========================================================
# DATA LOADING PARAMETERS
# =========================================================
# COLUMN names in dataset (input text and target labels).
TEXT_COLUMN = "source_text"
PRIVACY_COLUMN = "privacy"

RANDOM_STATE = 42

# =========================================================
# DATA SPLIT PARAMETERS
# =========================================================
# TEST_SIZE: Fraction of data to reserve for testing (0.15 = 15%)
TEST_SIZE = 0.15

# VAL_SIZE: Fraction of training data to use for validation (0.15 = 15%)
VAL_SIZE = 0.15

TARGET_LABELS = [
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
]

# Label name mapping for handling dataset variations
# If the dataset uses alternative label names, they are mapped to TARGET_LABELS here
LABEL_ALIASES = {
    "O": "O",
    "API_KEY": "API_KEY",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT_NUMBER",
    "IBAN": "IBAN",
    "PASSWORD": "PASSWORD",
    "SSN": "SSN",
    "FULL_NAME": "FULL_NAME",
    "FIRST_NAME": "FIRST_NAME",
    "LAST_NAME": "LAST_NAME",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PHONE": "PHONE_NUMBER",
}

# =========================================================
# CHARACTER N-GRAM VECTORIZER PARAMETERS
# =========================================================
# CHAR_NGRAM_RANGE: Character n-gram sizes (min, max)
#    - (3, 5) characters capture PII sub-patterns like "123" (SSN parts), "+45" (phones), or "@gm" (emails). Alternative could be (2, 6) for maximum morphological coverage, but drastically increases RAM usage.
CHAR_NGRAM_RANGE = (3, 5)

# CHAR_MAX_FEATURES: Maximum number of character n-gram features to extract
#    - Higher values = more features = potentially better but slower
#    - Tune: Increase for better performance (if memory/speed allows), decrease to speed up
#    - If None: High risk of Out-Of-Memory (OOM) crashes and overfitting, as 3-5 char n-gram combinations scale exponentially.
#    - Middle ground: 10000-15000 caps RAM usage while capturing enough PII variety. Alternatively, use SelectKBest post-vectorization. Best practice is to leave max_features high (e.g., 20000) but immediately funnel the output through Scikit-learn's SelectKBest(chi2, k=10000), which statistically prunes useless n-grams before training.grams the most informative features rather than just taking the most frequent ones.
CHAR_MAX_FEATURES = 8000

# CHAR_MIN_DF: Minimum document frequency for character n-grams
#    - Only include n-grams that appear in at least this many documents
#    - Filters out rare character patterns (noise)
#    - Tune: Increase to focus on common patterns, decrease to include rare patterns
CHAR_MIN_DF = 3

# =========================================================
# WORD N-GRAM VECTORIZER PARAMETERS
# =========================================================
# WORD_NGRAM_RANGE: Word n-gram sizes (min, max)
#    - (1, 2) captures single words and word pairs (e.g., "email", "email address")
#    - Tune: Increase for longer phrases, decrease for single words only
WORD_NGRAM_RANGE = (1, 2)

# WORD_MAX_FEATURES: Maximum number of word features to extract
#    - Higher values = more features = potentially better but slower
#    - Tune: Increase for better performance (if memory/speed allows), decrease to speed up
WORD_MAX_FEATURES = 6000

# WORD_MIN_DF: Minimum document frequency for words
#    - Only include words that appear in at least this many documents
#    - Filters out rare words (noise)
#    - Tune: Increase to focus on common words, decrease to include rare words
WORD_MIN_DF = 3

# WORD_MAX_DF: Maximum document frequency for words (as fraction, e.g., 0.95 = 95%)
#    - Exclude words that appear in more than this fraction of documents
#    - Filters out stop words that appear everywhere (less informative)
#    - Tune: Decrease to filter out more common words, increase to include more
WORD_MAX_DF = 0.95

# =========================================================
# SGD MODEL HYPERPARAMETERS
# =========================================================
# SGD_LOSS: Loss function for SGDClassifier
#    - "hinge": SVM-like loss (max-margin)
#    - "log_loss": Logistic regression (smooth)
#    - "modified_huber": Robust to outliers (good for noisy data)
#    - Tune: Try different losses to see which converges best for your data
SGD_LOSS = "modified_huber"

# SGD_ALPHA: Regularization strength (inverse of regularization parameter)
#    - Lower values = stronger regularization (simpler model, less overfitting)
#    - Higher values = weaker regularization (more complex model, more overfitting)
#    - Typical range: 1e-5 to 1e-2
#    - Tune: Increase if underfitting, decrease if overfitting
SGD_ALPHA = 1e-4

# SGD_PENALTY: Regularization type
#    - "l2": Ridge regularization (default, smooth penalty)
#    - "l1": Lasso regularization (produces sparsity)
#    - Tune: Use "l2" for dense models, "l1" for automatic feature selection
SGD_PENALTY = "l2"

# SGD_MAX_ITER: Maximum number of iterations for SGD solver
#    - Higher values = more iterations for convergence, slower training
#    - Lower values = faster training but may not converge
#    - Tune: Increase if solver doesn't converge, decrease for faster training
SGD_MAX_ITER = 2000

# SGD_TOL: Tolerance for stopping criterion (convergence threshold)
#    - If loss doesn't improve by at least this amount, training stops
#    - Smaller values = more iterations, better convergence
#    - Typical range: 1e-4 to 1e-2
#    - Tune: Decrease for more rigorous convergence, increase for faster training
SGD_TOL = 1e-3

# SGD_CLASS_WEIGHT: Class weight strategy for imbalanced datasets
#    - "balanced": Automatically adjust weights inversely to class frequency
#    - None: All classes weighted equally
#    - Tune: Use "balanced" if label distribution is highly skewed
SGD_CLASS_WEIGHT = "balanced"

# SGD_N_JOBS: Number of parallel jobs for OneVsRestClassifier
#    - -1 = use all CPU cores (fast but uses more memory)
#    - 1 = single-threaded (slower but stable, recommended for Windows)
#    - Tune: Use -1 for fast multi-core systems, 1 for stable single-core
SGD_N_JOBS = -1

# =========================================================
# Start rimeligt small så jobbet ikke dør som en klovn

# HYPERPARAMETER SEARCH SPACE
# Grid of parameter combinations to try during hyperparameter search
# Kept small to avoid excessive training time
# =========================================================
SEARCH_CHAR_MAX_FEATURES = [4000, 8000]
SEARCH_WORD_MAX_FEATURES = [3000, 6000]
SEARCH_CHAR_NGRAM_RANGE = [(3, 5)]
SEARCH_WORD_NGRAM_RANGE = [(1, 2)]
SEARCH_ALPHA = [1e-5, 5e-5, 1e-4]
SEARCH_LOSS = ["hinge"]

# =========================================================
# DISPLAY AND OUTPUT PARAMETERS
# =========================================================
# MAX_TEXT_PREVIEW: Maximum characters to show from a text example in logs
#    - Higher values = more text context visible
#    - Tune: Increase to see more text, decrease for condensed output
MAX_TEXT_PREVIEW = 400

# NUM_EXAMPLES_TO_SHOW: Number of prediction examples to display after evaluation
#    - Tune: Increase to see more examples, decrease for faster output
NUM_EXAMPLES_TO_SHOW = 5


# =========================================================
# HJÆLPEFUNKTIONER
# =========================================================
def short_text(text, max_len=400):
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def decode_multilabel_row(binary_row, classes):
    return [classes[i] for i, value in enumerate(binary_row) if value == 1]


def extract_labels_from_privacy(value):
    """
    Extracts and normalizes PII labels from the dataset's 'privacy' column.
    The 'privacy' column often contains lists of dictionaries, sometimes as raw strings.
    """
    labels = []

    # Safely parse stringified lists (e.g., "[{'label': 'EMAIL'}]") back into Python objects
    # Maybe remove the *ast???
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        value = []

    # Iterate through the extracted entities to find and map priority labels
    for item in value:
        if isinstance(item, dict):
            raw_label = str(item.get("label", "")).strip().upper()
            # Unify different naming conventions (e.g., 'EMAIL_ADDRESS' -> 'EMAIL')
            mapped_label = LABEL_ALIASES.get(raw_label)

            if mapped_label in TARGET_LABELS and mapped_label != "O":
                labels.append(mapped_label)

    # If no priority labels are found, mark the text as 'O' (Outside/No PII)
    if len(labels) == 0:
        labels = ["O"]

    # Return unique labels only to represent a multi-label classification target
    return list(dict.fromkeys(labels))


# =========================================================
# DATA
# =========================================================
def load_data():
    df = load_combined_dataset()

    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Antal rækker: {len(df)}")
    print(f"Kolonner: {list(df.columns)}\n")

    return df


def inspect_data(df):
    print("=" * 80)
    print("KORT PREVIEW")
    print("=" * 80)
    print(df.head(5).to_string())
    print()


def clean_data(df):
    df = df[[TEXT_COLUMN, PRIVACY_COLUMN]].copy()
    df = df.dropna(subset=[TEXT_COLUMN, PRIVACY_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    df["all_labels"] = df[PRIVACY_COLUMN].apply(extract_labels_from_privacy)
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    print("=" * 80)
    print("EFTER RENSNING")
    print("=" * 80)
    print(f"Antal rækker tilbage: {len(df)}")

    label_counter = {}
    for labels in df["all_labels"]:
        for label in labels:
            label_counter[label] = label_counter.get(label, 0) + 1

    print("\nLabelfordeling:")
    print(pd.Series(label_counter).sort_values(ascending=False).to_string())
    print()

    return df[[TEXT_COLUMN, "all_labels"]]


def split_data(df):
    X, y = df[TEXT_COLUMN], df["all_labels"]

    # Step 1: Hold out a completely unseen test set (e.g., 15%) from the main data.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 2: Split the remaining data into train and validation sets.
    # We adjust the relative val_size so validation remains exactly VAL_SIZE of the entire dataset.
    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=RANDOM_STATE
    )

    print("=" * 80)
    print("DATA SPLIT")
    print("=" * 80)
    print(f"Træning:    {len(X_train)} eksempler")
    print(f"Validering: {len(X_val)} eksempler")
    print(f"Test:       {len(X_test)} eksempler\n")

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train,
        y_val,
        y_test,
    )


def prepare_targets(y_train, y_val, y_test):
    mlb = MultiLabelBinarizer(classes=TARGET_LABELS)
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin = mlb.transform(y_val)
    y_test_bin = mlb.transform(y_test)

    print("=" * 80)
    print("MULTILABEL TARGET FORMAT")
    print("=" * 80)
    print(f"Klasser: {list(mlb.classes_)}")
    print(f"y_train_bin shape: {y_train_bin.shape}\n")

    return mlb, y_train_bin, y_val_bin, y_test_bin


# =========================================================
# FEATURES
# =========================================================
def build_vectorizers(params):
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=params["char_ngram_range"],
        max_features=params["char_max_features"],
        min_df=CHAR_MIN_DF,
        sublinear_tf=True,
    )

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=params["word_ngram_range"],
        max_features=params["word_max_features"],
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        sublinear_tf=True,
    )

    return char_vectorizer, word_vectorizer


def fit_transform_features(X_train, X_val, X_test, params):
    """
    Transforms raw text data into sparse numerical TF-IDF feature matrices.
    """
    char_vectorizer, word_vectorizer = build_vectorizers(params)

    # Fit (learn vocabulary) ONLY on training data, transform train, val, and test to prevent data leakage.
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)
    X_test_word = word_vectorizer.transform(X_test)

    # Combine character and word features horizontally into single sparse matrices per split.
    # .tocsr() ensures memory-efficient Compressed Sparse Row format for faster Scikit-learn operations.
    X_train_all = hstack([X_train_char, X_train_word]).tocsr()
    X_val_all = hstack([X_val_char, X_val_word]).tocsr()
    X_test_all = hstack([X_test_char, X_test_word]).tocsr()

    return char_vectorizer, word_vectorizer, X_train_all, X_val_all, X_test_all


# =========================================================
# MODEL
# =========================================================
def build_model(params):
    """
    Constructs an SVM model wrapped in a OneVsRestClassifier for multi-label prediction.
    """
    # Base model is an SVM (via SGD) optimized specifically for sparse, high-dimensional text data.
    base_model = SGDClassifier(
        loss=params["loss"],
        alpha=params["alpha"],
        penalty=SGD_PENALTY,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
        random_state=RANDOM_STATE,
        class_weight=SGD_CLASS_WEIGHT,  # Corrects for label imbalance.
        n_jobs=SGD_N_JOBS,
    )

    # Since standard SGDClassifier only handles single labels, we wrap it to train one model per PII class.
    model = OneVsRestClassifier(base_model, n_jobs=SGD_N_JOBS)
    return model


def train_model(X_train_features, y_train_bin, params):
    model = build_model(params)
    model.fit(X_train_features, y_train_bin)
    return model


# =========================================================
# HYPERPARAMETER SEARCH
# =========================================================
def build_search_candidates():
    candidates = []

    for (
        char_max_features,
        word_max_features,
        char_ngram_range,
        word_ngram_range,
        alpha,
        loss,
    ) in itertools.product(
        SEARCH_CHAR_MAX_FEATURES,
        SEARCH_WORD_MAX_FEATURES,
        SEARCH_CHAR_NGRAM_RANGE,
        SEARCH_WORD_NGRAM_RANGE,
        SEARCH_ALPHA,
        SEARCH_LOSS,
    ):
        candidates.append(
            {
                "char_max_features": char_max_features,
                "word_max_features": word_max_features,
                "char_ngram_range": char_ngram_range,
                "word_ngram_range": word_ngram_range,
                "alpha": alpha,
                "loss": loss,
            }
        )

    return candidates


def hyperparameter_search(X_train, y_train_bin, X_val, y_val_bin):
    candidates = build_search_candidates()

    print("=" * 80)
    print("HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Antal kombinationer: {len(candidates)}\n")

    best_params = None
    best_model = None
    best_char_vectorizer = None
    best_word_vectorizer = None
    best_val_macro_f1 = -1.0

    for idx, params in enumerate(candidates, start=1):
        print(f"Kombi {idx}/{len(candidates)}")
        print(params)

        char_vectorizer, word_vectorizer, X_train_feat, X_val_feat, _ = (
            fit_transform_features(X_train, X_val, X_val, params)
        )

        model = train_model(X_train_feat, y_train_bin, params)
        y_val_pred = model.predict(X_val_feat)

        val_macro_f1 = f1_score(y_val_bin, y_val_pred, average="macro", zero_division=0)
        val_micro_f1 = f1_score(y_val_bin, y_val_pred, average="micro", zero_division=0)
        val_subset_acc = accuracy_score(y_val_bin, y_val_pred)

        print(f"  val macro F1: {val_macro_f1:.4f}")
        print(f"  val micro F1: {val_micro_f1:.4f}")
        print(f"  val accuracy: {val_subset_acc:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_params = params
            best_model = model
            best_char_vectorizer = char_vectorizer
            best_word_vectorizer = word_vectorizer
            print("  -> Ny bedste kombination")

        print()

    print("=" * 80)
    print("BEDSTE HYPERPARAMETRE")
    print("=" * 80)
    print(best_params)
    print(f"Bedste val macro F1: {best_val_macro_f1:.4f}\n")

    return best_params, best_model, best_char_vectorizer, best_word_vectorizer


# =========================================================
# EVALUERING
# =========================================================
def evaluate_model(model, X_test_features, y_test_bin, mlb):
    print("=" * 80)
    print("EVALUERING PÅ TESTDATA")
    print("=" * 80)

    y_pred_bin = model.predict(X_test_features)

    subset_acc = accuracy_score(y_test_bin, y_pred_bin)
    micro_f1 = f1_score(y_test_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test_bin, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0)
    ham_loss = hamming_loss(y_test_bin, y_pred_bin)

    print(
        classification_report(
            y_test_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0
        )
    )
    print(f"accuracy:     {subset_acc:.4f}")
    print(f"macro avg f1: {macro_f1:.4f}")
    print(f"weighted f1:  {weighted_f1:.4f}")
    print(f"micro f1:     {micro_f1:.4f}")
    print(f"hamming loss: {ham_loss:.4f}\n")

    print("=" * 80)
    print("CONFUSION MATRIX PR. LABEL")
    print("=" * 80)

    mcm = multilabel_confusion_matrix(y_test_bin, y_pred_bin)
    for idx, label in enumerate(mlb.classes_):
        tn, fp, fn, tp = mcm[idx].ravel()
        print(f"{label}: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print()

    return y_pred_bin


def show_predictions(X_test, y_test_bin, y_pred_bin, mlb, n=5):
    print("=" * 100)
    print("EKSEMPLER PÅ PREDICTIONS")
    print("=" * 100)

    for i in range(min(n, len(X_test))):
        actual = set(decode_multilabel_row(y_test_bin[i], mlb.classes_))
        predicted = set(decode_multilabel_row(y_pred_bin[i], mlb.classes_))

        print(f"\nEksempel {i + 1}:")
        print(f"Actual:    {sorted(actual)}")
        print(f"Predicted: {sorted(predicted)}")
        print(f"Korrekt:   {'JA' if actual == predicted else 'NEJ'}")
        print(f"Mangler:   {sorted(actual - predicted) or 'Ingen'}")
        print(f"Ekstra:    {sorted(predicted - actual) or 'Ingen'}")
        print(f"Tekst:     {short_text(X_test.iloc[i], MAX_TEXT_PREVIEW)}")
        print("-" * 100)


# =========================================================
# GEM MODEL
# =========================================================
def save_artifacts(model, char_vectorizer, word_vectorizer, mlb, params):
    os.makedirs("saved_model", exist_ok=True)

    joblib.dump(model, "saved_model/model.joblib")
    joblib.dump(char_vectorizer, "saved_model/char_vectorizer.joblib")
    joblib.dump(word_vectorizer, "saved_model/word_vectorizer.joblib")
    joblib.dump(mlb, "saved_model/mlb.joblib")
    joblib.dump(params, "saved_model/best_params.joblib")

    print("Gemte model artifacts i mappen: saved_model/\n")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data()
    inspect_data(df)
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    mlb, y_train_bin, y_val_bin, y_test_bin = prepare_targets(y_train, y_val, y_test)

    best_params, _, best_char_vectorizer, best_word_vectorizer = hyperparameter_search(
        X_train, y_train_bin, X_val, y_val_bin
    )

    # Refit the final vectorizers & model on *both* train and validation partitions.
    # The validation set was used for hyperparameter tuning; now include it to maximize the final training data.
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval_bin = np.vstack([y_train_bin, y_val_bin])

    X_trainval_char = best_char_vectorizer.fit_transform(X_trainval)
    X_test_char = best_char_vectorizer.transform(X_test)

    X_trainval_word = best_word_vectorizer.fit_transform(X_trainval)
    X_test_word = best_word_vectorizer.transform(X_test)

    X_trainval_feat = hstack([X_trainval_char, X_trainval_word]).tocsr()
    X_test_feat = hstack([X_test_char, X_test_word]).tocsr()

    final_model = train_model(X_trainval_feat, y_trainval_bin, best_params)
    y_pred_bin = evaluate_model(final_model, X_test_feat, y_test_bin, mlb)
    show_predictions(X_test, y_test_bin, y_pred_bin, mlb, n=NUM_EXAMPLES_TO_SHOW)

    save_artifacts(
        model=final_model,
        char_vectorizer=best_char_vectorizer,
        word_vectorizer=best_word_vectorizer,
        mlb=mlb,
        params=best_params,
    )


if __name__ == "__main__":
    main()
