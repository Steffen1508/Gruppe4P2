import os

# =========================================================
# WINDOWS STABILITET / MINDRE RISIKO FOR CRASH
# Skal stå før sklearn/numpy/scipy importeres
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ast
import pandas as pd
import numpy as np

from chunking import token_chunking

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix
)

# =========================================================
# INDSTILLINGER 
# =========================================================
FILE_PATH = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"
TEXT_COLUMN = "source_text"
PRIVACY_COLUMN = "privacy"

RANDOM_STATE = 42

# split: train / val / test
TEST_SIZE = 0.15
VAL_SIZE = 0.15

TARGET_LABELS = [
    "EMAIL",
    "PHONE_NUMBER",
    "FIRST_NAME",
    "LAST_NAME",
    "USERNAME",
    "CITY",
    "STREET",
    "ZIPCODE"
]

NUM_EXAMPLES_TO_SHOW = 8
MAX_TEXT_PREVIEW = 500

# vectorizer settings
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MAX_FEATURES = 12000
CHAR_MIN_DF = 3

WORD_NGRAM_RANGE = (1, 2)
WORD_MAX_FEATURES = 8000
WORD_MIN_DF = 3
WORD_MAX_DF = 0.95

# threshold tuning på decision scores
THRESHOLD_GRID = np.arange(-1.5, 1.51, 0.1)

# model
SVM_C = 1.0
SVM_MAX_ITER = 3000

# chunking settings (træningstekster splittes i chunks)
ENABLE_TEXT_CHUNKING = True
CHUNK_SIZE = 3
CHUNK_OVERLAP = 1
CHUNK_MAX_TOKENS = 64


# =========================================================
# LABEL HJÆLP
# =========================================================
def extract_labels_from_privacy(value):
    labels = []

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return labels

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return labels

    for item in value:
        if isinstance(item, dict):
            label = item.get("label")
            if label in TARGET_LABELS:
                labels.append(label)

    # fjern dubletter, behold rækkefølge
    return list(dict.fromkeys(labels))


def decode_multilabel_row(binary_row, classes):
    return [classes[i] for i, value in enumerate(binary_row) if value == 1]


def short_text(text, max_len=500):
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def chunk_text(text):
    text = str(text)

    if not ENABLE_TEXT_CHUNKING:
        return [text]

    chunks = token_chunking(
        text=text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        max_tokens=CHUNK_MAX_TOKENS
    )

    if not chunks:
        return [text]

    return chunks


def expand_training_with_chunks(X_train, y_train_bin):
    X_train = X_train.reset_index(drop=True)

    expanded_texts = []
    expanded_targets = []

    for i, text in enumerate(X_train):
        chunks = chunk_text(text)
        for chunk in chunks:
            expanded_texts.append(chunk)
            expanded_targets.append(y_train_bin[i])

    expanded_texts = pd.Series(expanded_texts)
    expanded_targets = np.asarray(expanded_targets)

    print("=" * 80)
    print("CHUNKING AF TRÆNINGSDATA")
    print("=" * 80)
    print(f"Originale træningstekster: {len(X_train)}")
    print(f"Udvidede træningschunks:   {len(expanded_texts)}")
    print(f"Gns. chunks pr. tekst:     {len(expanded_texts) / max(len(X_train), 1):.2f}")
    print()

    return expanded_texts, expanded_targets


# =========================================================
# DATA
# =========================================================
def load_data(file_path):
    df = pd.read_parquet(file_path)

    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"Antal rækker: {len(df)}")
    print(f"Antal kolonner: {len(df.columns)}")
    print("\nKolonner:")
    for col in df.columns:
        print(f" - {col}")
    print()

    return df


def inspect_data(df):
    print("=" * 80)
    print("KORT PREVIEW")
    print("=" * 80)
    print(df.head(5).to_string())
    print()

    print("=" * 80)
    print("INFO OM LABELS")
    print("=" * 80)
    print("Rå labels ligger i kolonnen 'privacy' som struktureret data.")
    print("Alle relevante labels bliver udledt i clean_data().")
    print()


def clean_data(df):
    df = df[[TEXT_COLUMN, PRIVACY_COLUMN]].copy()
    df = df.dropna(subset=[TEXT_COLUMN, PRIVACY_COLUMN])

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df["all_labels"] = df[PRIVACY_COLUMN].apply(extract_labels_from_privacy)

    df = df[df["all_labels"].apply(len) > 0]
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    print("=" * 80)
    print("EFTER RENSNING")
    print("=" * 80)
    print(f"Antal rækker tilbage: {len(df)}")
    print()

    print("Fordeling af labels på tværs af alle rækker:")
    label_counter = {}
    for labels in df["all_labels"]:
        for label in labels:
            label_counter[label] = label_counter.get(label, 0) + 1

    label_counts_series = pd.Series(label_counter).sort_values(ascending=False)
    print(label_counts_series.to_string())
    print()

    print("Antal labels per tekst:")
    print(df["all_labels"].apply(len).value_counts().sort_index().to_string())
    print()

    return df[[TEXT_COLUMN, "all_labels"]]


def split_data(df):
    X = df[TEXT_COLUMN]
    y = df["all_labels"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=RANDOM_STATE
    )

    print("=" * 80)
    print("TRAIN / VAL / TEST SPLIT")
    print("=" * 80)
    print(f"Train størrelse: {len(X_train)}")
    print(f"Val størrelse:   {len(X_val)}")
    print(f"Test størrelse:  {len(X_test)}")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_targets(y_train, y_val, y_test):
    mlb = MultiLabelBinarizer(classes=TARGET_LABELS)

    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin = mlb.transform(y_val)
    y_test_bin = mlb.transform(y_test)

    print("=" * 80)
    print("MULTILABEL TARGET FORMAT")
    print("=" * 80)
    print("Klasser i rækkefølge:")
    print(list(mlb.classes_))
    print()
    print(f"y_train_bin shape: {y_train_bin.shape}")
    print(f"y_val_bin shape:   {y_val_bin.shape}")
    print(f"y_test_bin shape:  {y_test_bin.shape}")
    print()

    return mlb, y_train_bin, y_val_bin, y_test_bin


# =========================================================
# FEATURES
# =========================================================
def build_vectorizers():
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=CHAR_MAX_FEATURES,
        min_df=CHAR_MIN_DF,
        sublinear_tf=True
    )

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        max_features=WORD_MAX_FEATURES,
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        sublinear_tf=True
    )

    return char_vectorizer, word_vectorizer


def fit_transform_features(X_train, X_val, X_test):
    print("=" * 80)
    print("BYGGER FEATURES")
    print("=" * 80)

    char_vectorizer, word_vectorizer = build_vectorizers()

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_all = hstack([X_train_char, X_train_word]).tocsr()
    X_val_all = hstack([X_val_char, X_val_word]).tocsr()
    X_test_all = hstack([X_test_char, X_test_word]).tocsr()

    print(f"Train feature shape: {X_train_all.shape}")
    print(f"Val feature shape:   {X_val_all.shape}")
    print(f"Test feature shape:  {X_test_all.shape}")
    print()

    return (
        char_vectorizer,
        word_vectorizer,
        X_train_all,
        X_val_all,
        X_test_all
    )


# =========================================================
# MODEL
# =========================================================
def train_model(X_train_features, y_train_bin):
    print("=" * 80)
    print("TRÆNER TF IDF + SVM MODEL")
    print("=" * 80)

    base_model = LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE
    )

    # n_jobs=1 for at undgå Windows crash
    model = OneVsRestClassifier(base_model, n_jobs=1)
    model.fit(X_train_features, y_train_bin)

    print("Model trænet færdig.\n")
    return model


# =========================================================
# THRESHOLD TUNING PÅ DECISION SCORES
# =========================================================
def ensure_2d_scores(score_matrix):
    score_matrix = np.asarray(score_matrix)
    if score_matrix.ndim == 1:
        score_matrix = score_matrix.reshape(-1, 1)
    return score_matrix


def tune_thresholds(model, X_val_features, y_val_bin, mlb):
    print("=" * 80)
    print("TUNER THRESHOLDS PR. LABEL")
    print("=" * 80)

    val_scores = model.decision_function(X_val_features)
    val_scores = ensure_2d_scores(val_scores)

    thresholds = {}

    for i, label in enumerate(mlb.classes_):
        best_threshold = 0.0
        best_f1 = -1.0

        y_true = y_val_bin[:, i]

        for threshold in THRESHOLD_GRID:
            y_pred = (val_scores[:, i] >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        thresholds[label] = best_threshold
        print(f"{label:<15} best_threshold={best_threshold:.2f}  val_f1={best_f1:.4f}")

    print()
    return thresholds


def apply_thresholds(score_matrix, thresholds, mlb):
    score_matrix = ensure_2d_scores(score_matrix)
    pred_bin = np.zeros_like(score_matrix, dtype=int)

    for i, label in enumerate(mlb.classes_):
        threshold = thresholds[label]
        pred_bin[:, i] = (score_matrix[:, i] >= threshold).astype(int)

    return pred_bin


# =========================================================
# EVALUERING
# =========================================================
def evaluate_model(model, X_test_features, y_test_bin, mlb, thresholds):
    print("=" * 80)
    print("EVALUERING")
    print("=" * 80)

    test_scores = model.decision_function(X_test_features)
    test_scores = ensure_2d_scores(test_scores)

    y_pred_bin = apply_thresholds(test_scores, thresholds, mlb)

    subset_acc = accuracy_score(y_test_bin, y_pred_bin)
    micro_f1 = f1_score(y_test_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test_bin, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0)
    ham_loss = hamming_loss(y_test_bin, y_pred_bin)

    print(f"Subset accuracy: {subset_acc:.4f}")
    print("  Andel af tekster hvor ALLE labels rammes perfekt.")
    print()
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    print(f"Hamming loss:    {ham_loss:.4f}")
    print()

    print("Classification report:")
    print(classification_report(
        y_test_bin,
        y_pred_bin,
        target_names=mlb.classes_,
        zero_division=0
    ))
    print()

    print("=" * 80)
    print("CONFUSION MATRIX PR. LABEL")
    print("=" * 80)

    mcm = multilabel_confusion_matrix(y_test_bin, y_pred_bin)
    for idx, label in enumerate(mlb.classes_):
        tn, fp, fn, tp = mcm[idx].ravel()
        print(f"{label}:")
        print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print()

    return y_pred_bin, test_scores


# =========================================================
# OUTPUT EKSEMPLER
# =========================================================
def show_predictions(X_test, y_test_bin, y_pred_bin, test_scores, mlb, thresholds, n=8):
    print("=" * 100)
    print("EKSEMPLER PÅ MULTILABEL FORUDSIGELSER")
    print("=" * 100)

    X_test = X_test.reset_index(drop=True)

    for i in range(min(n, len(X_test))):
        text = X_test.iloc[i]
        actual_labels = decode_multilabel_row(y_test_bin[i], mlb.classes_)
        predicted_labels = decode_multilabel_row(y_pred_bin[i], mlb.classes_)

        actual_set = set(actual_labels)
        predicted_set = set(predicted_labels)

        missing_labels = sorted(actual_set - predicted_set)
        extra_labels = sorted(predicted_set - actual_set)
        correct = actual_set == predicted_set

        print(f"\nEksempel {i + 1}")
        print("-" * 100)
        print(f"Actual labels:    {actual_labels}")
        print(f"Predicted labels: {predicted_labels}")
        print(f"Korrekt samlet:   {'JA' if correct else 'NEJ'}")
        print(f"Manglende labels: {missing_labels if missing_labels else 'Ingen'}")
        print(f"Ekstra labels:    {extra_labels if extra_labels else 'Ingen'}")

        print("\nLabel scores:")
        for j, label in enumerate(mlb.classes_):
            score = float(test_scores[i, j])
            thr = thresholds[label]
            mark = "X" if score >= thr else "-"
            print(f"  {mark} {label:<15} score={score:.3f}  threshold={thr:.2f}")

        print("\nTekst:")
        print(short_text(text, MAX_TEXT_PREVIEW))
        print("-" * 100)


def show_error_examples(X_test, y_test_bin, y_pred_bin, mlb, n=5):
    print("=" * 100)
    print("FORKERTE EKSEMPLER")
    print("=" * 100)

    X_test = X_test.reset_index(drop=True)

    shown = 0
    for i in range(len(X_test)):
        actual_labels = set(decode_multilabel_row(y_test_bin[i], mlb.classes_))
        predicted_labels = set(decode_multilabel_row(y_pred_bin[i], mlb.classes_))

        if actual_labels != predicted_labels:
            missing_labels = sorted(actual_labels - predicted_labels)
            extra_labels = sorted(predicted_labels - actual_labels)

            print(f"\nEksempel {i + 1}")
            print("-" * 100)
            print(f"Actual labels:    {sorted(actual_labels)}")
            print(f"Predicted labels: {sorted(predicted_labels)}")
            print(f"Manglende labels: {missing_labels if missing_labels else 'Ingen'}")
            print(f"Ekstra labels:    {extra_labels if extra_labels else 'Ingen'}")
            print("\nTekst:")
            print(short_text(X_test.iloc[i], MAX_TEXT_PREVIEW))
            print("-" * 100)

            shown += 1
            if shown >= n:
                break

    if shown == 0:
        print("Ingen forkerte eksempler fundet i det udvalgte udsnit.\n")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data(FILE_PATH)
    inspect_data(df)
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    mlb, y_train_bin, y_val_bin, y_test_bin = prepare_targets(y_train, y_val, y_test)

    X_train_for_features, y_train_for_model = expand_training_with_chunks(X_train, y_train_bin)

    (
        char_vectorizer,
        word_vectorizer,
        X_train_features,
        X_val_features,
        X_test_features
    ) = fit_transform_features(X_train_for_features, X_val, X_test)

    model = train_model(X_train_features, y_train_for_model)
    thresholds = tune_thresholds(model, X_val_features, y_val_bin, mlb)

    y_pred_bin, test_scores = evaluate_model(
        model=model,
        X_test_features=X_test_features,
        y_test_bin=y_test_bin,
        mlb=mlb,
        thresholds=thresholds
    )

    show_predictions(
        X_test=X_test,
        y_test_bin=y_test_bin,
        y_pred_bin=y_pred_bin,
        test_scores=test_scores,
        mlb=mlb,
        thresholds=thresholds,
        n=NUM_EXAMPLES_TO_SHOW
    )

    show_error_examples(
        X_test=X_test,
        y_test_bin=y_test_bin,
        y_pred_bin=y_pred_bin,
        mlb=mlb,
        n=5
    )


if __name__ == "__main__":
    main()