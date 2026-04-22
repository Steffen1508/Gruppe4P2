import os
import sys
import ast
import time
import itertools
import warnings

import joblib
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

# Gets the absolute path of this file, then navigates up two levels to find the project root folder.
# Adds the root directory to Python's path so we can import scripts (like data_loader.py) from outside this folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_combined_dataset

warnings.filterwarnings("ignore")

# =========================================================
# CPU / THREAD SETTINGS
# =========================================================
# Fine at keep low if your machine / server becomes unstable.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =========================================================
# DATA LOADING PARAMETERS
# =========================================================
TEXT_COLUMN = "source_text"
PRIVACY_COLUMN = "privacy"
RANDOM_STATE = 42

# =========================================================
# DATA SPLIT PARAMETERS
# =========================================================
TEST_SIZE = 0.15
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
# DEFAULT BASELINE PARAMETERS
# Used if you want a single non-search run later
# =========================================================
BASE_PARAMS = {
    "char_ngram_range": (3, 5),
    "char_max_features": 8000,
    "char_min_df": 0.0001,
    "word_ngram_range": (1, 2),
    "word_max_features": 6000,
    "word_min_df": 50,
    "word_max_df": 0.60,
    "loss": "hinge",
    "alpha": 1e-4,
    "penalty": "l2",
    "max_iter": 2000,
    "tol": 1e-3,
    "class_weight": "balanced",
}

# =========================================================
# GRID SEARCH SPACE
# Keep this realistic. Otherwise the job explodes.
# =========================================================
SEARCH_GRID = {
    "char_ngram_range": [(3, 5), (3, 6), (2, 6)],
    "char_max_features": [100000, 200000],
    "char_min_df": [2, 5, 0.0001],
    "word_ngram_range": [(1, 2), (1, 3)],
    "word_max_features": [60000, 100000],
    "word_min_df": [2, 5, 10],
    "word_max_df": [0.40, 0.60, 0.80],
    "loss": ["hinge", "modified_huber"],
    "alpha": [1e-6, 1e-5, 1e-4, 1e-3],
    "penalty": ["l2"],
    "max_iter": [3000, 8000],
    "tol": [1e-4, 1e-5],
    "class_weight": ["balanced"],
}

# If you want to avoid 1000+ combinations on a big dataset, cap it here.
# Set to None to run all combinations.
MAX_SEARCH_COMBINATIONS = 200

# Windows stability: often safer with 1
OVR_N_JOBS = 1

# =========================================================
# DISPLAY AND OUTPUT PARAMETERS
# =========================================================
MAX_TEXT_PREVIEW = 400
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

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        value = []

    for item in value:
        if isinstance(item, dict):
            raw_label = str(item.get("label", "")).strip().upper()
            mapped_label = LABEL_ALIASES.get(raw_label)

            if mapped_label in TARGET_LABELS and mapped_label != "O":
                labels.append(mapped_label)

    if len(labels) == 0:
        labels = ["O"]

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

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

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
        min_df=params["char_min_df"],
        sublinear_tf=True,
    )

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=params["word_ngram_range"],
        max_features=params["word_max_features"],
        min_df=params["word_min_df"],
        max_df=params["word_max_df"],
        sublinear_tf=True,
    )

    return char_vectorizer, word_vectorizer


def fit_transform_features(X_train, X_val, X_test, params):
    """
    Transforms raw text data into sparse numerical TF-IDF feature matrices.
    """
    char_vectorizer, word_vectorizer = build_vectorizers(params)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_all = hstack([X_train_char, X_train_word]).tocsr()
    X_val_all = hstack([X_val_char, X_val_word]).tocsr()
    X_test_all = hstack([X_test_char, X_test_word]).tocsr()

    return char_vectorizer, word_vectorizer, X_train_all, X_val_all, X_test_all


# =========================================================
# MODEL
# =========================================================
def build_model(params):
    """
    Constructs an SGD-based linear classifier wrapped in OneVsRest for multilabel prediction.
    """
    base_model = SGDClassifier(
        loss=params["loss"],
        alpha=params["alpha"],
        penalty=params["penalty"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        random_state=RANDOM_STATE,
        class_weight=params["class_weight"],
    )

    model = OneVsRestClassifier(base_model, n_jobs=OVR_N_JOBS)
    return model


def train_model(X_train_features, y_train_bin, params):
    model = build_model(params)
    model.fit(X_train_features, y_train_bin)
    return model


# =========================================================
# HYPERPARAMETER SEARCH
# =========================================================
def build_search_candidates():
    keys = list(SEARCH_GRID.keys())
    values = [SEARCH_GRID[key] for key in keys]

    candidates = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        candidates.append(params)

    if MAX_SEARCH_COMBINATIONS is not None and len(candidates) > MAX_SEARCH_COMBINATIONS:
        candidates = candidates[:MAX_SEARCH_COMBINATIONS]

    return candidates


def hyperparameter_search(X_train, y_train_bin, X_val, y_val_bin):
    candidates = build_search_candidates()

    print("=" * 80)
    print("HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Antal kombinationer der køres: {len(candidates)}\n")

    best_params = None
    best_model = None
    best_char_vectorizer = None
    best_word_vectorizer = None
    best_val_macro_f1 = -1.0

    results = []

    for idx, params in enumerate(candidates, start=1):
        print(f"Kombi {idx}/{len(candidates)}")
        print(params)

        start_time = time.time()

        try:
            char_vectorizer, word_vectorizer, X_train_feat, X_val_feat, _ = (
                fit_transform_features(X_train, X_val, X_val, params)
            )

            model = train_model(X_train_feat, y_train_bin, params)
            y_val_pred = model.predict(X_val_feat)

            val_macro_f1 = f1_score(y_val_bin, y_val_pred, average="macro", zero_division=0)
            val_micro_f1 = f1_score(y_val_bin, y_val_pred, average="micro", zero_division=0)
            val_subset_acc = accuracy_score(y_val_bin, y_val_pred)

            elapsed = time.time() - start_time

            result_row = {
                **params,
                "val_macro_f1": val_macro_f1,
                "val_micro_f1": val_micro_f1,
                "val_subset_acc": val_subset_acc,
                "seconds": round(elapsed, 2),
            }
            results.append(result_row)

            print(f"  val macro F1: {val_macro_f1:.4f}")
            print(f"  val micro F1: {val_micro_f1:.4f}")
            print(f"  val accuracy: {val_subset_acc:.4f}")
            print(f"  tid:          {elapsed:.2f} sek")

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_params = params
                best_model = model
                best_char_vectorizer = char_vectorizer
                best_word_vectorizer = word_vectorizer
                print("  -> Ny bedste kombination")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  FEJL: {e}")
            print(f"  tid:  {elapsed:.2f} sek")

            result_row = {
                **params,
                "val_macro_f1": np.nan,
                "val_micro_f1": np.nan,
                "val_subset_acc": np.nan,
                "seconds": round(elapsed, 2),
                "error": str(e),
            }
            results.append(result_row)

        print()

    results_df = pd.DataFrame(results).sort_values(
        by="val_macro_f1", ascending=False, na_position="last"
    )

    print("=" * 80)
    print("BEDSTE HYPERPARAMETRE")
    print("=" * 80)
    print(best_params)
    print(f"Bedste val macro F1: {best_val_macro_f1:.4f}\n")

    print("=" * 80)
    print("TOP 10 RESULTATER")
    print("=" * 80)
    display_cols = [
        "val_macro_f1",
        "val_micro_f1",
        "val_subset_acc",
        "seconds",
        "char_ngram_range",
        "char_max_features",
        "char_min_df",
        "word_ngram_range",
        "word_max_features",
        "word_min_df",
        "word_max_df",
        "loss",
        "alpha",
        "penalty",
        "max_iter",
        "tol",
        "class_weight",
    ]
    print(results_df[display_cols].head(10).to_string(index=False))
    print()

    return best_params, best_model, best_char_vectorizer, best_word_vectorizer, results_df


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
def save_artifacts(model, char_vectorizer, word_vectorizer, mlb, params, search_results_df):
    os.makedirs("saved_model", exist_ok=True)

    joblib.dump(model, "saved_model/model.joblib")
    joblib.dump(char_vectorizer, "saved_model/char_vectorizer.joblib")
    joblib.dump(word_vectorizer, "saved_model/word_vectorizer.joblib")
    joblib.dump(mlb, "saved_model/mlb.joblib")
    joblib.dump(params, "saved_model/best_params.joblib")
    search_results_df.to_csv("saved_model/grid_search_results.csv", index=False)

    print("Gemte model artifacts i mappen: saved_model/")
    print("Gemte også grid search resultater i: saved_model/grid_search_results.csv\n")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data()
    inspect_data(df)
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    mlb, y_train_bin, y_val_bin, y_test_bin = prepare_targets(y_train, y_val, y_test)

    (
        best_params,
        _,
        best_char_vectorizer,
        best_word_vectorizer,
        search_results_df,
    ) = hyperparameter_search(X_train, y_train_bin, X_val, y_val_bin)

    if best_params is None:
        raise RuntimeError("Ingen gyldig model blev trænet under grid search.")

    # Refit on train + validation
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
        search_results_df=search_results_df,
    )


if __name__ == "__main__":
    main()