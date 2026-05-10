import os
import sys
import ast
import joblib
import pandas as pd
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_combined_dataset


TEXT_COLUMN = "source_text"
PRIVACY_COLUMN = "privacy"

RANDOM_STATE = 42
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


def extract_labels_from_privacy(value):
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


def clean_data(df):
    df = df[[TEXT_COLUMN, PRIVACY_COLUMN]].copy()
    df = df.dropna(subset=[TEXT_COLUMN, PRIVACY_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df["all_labels"] = df[PRIVACY_COLUMN].apply(extract_labels_from_privacy)
    df = df[df[TEXT_COLUMN].str.strip() != ""]
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

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train,
        y_val,
        y_test,
    )


def main():
    print("=" * 80)
    print("LOADING SAVED MODEL")
    print("=" * 80)

    model = joblib.load("saved_model/model.joblib")
    char_vectorizer = joblib.load("saved_model/char_vectorizer.joblib")
    word_vectorizer = joblib.load("saved_model/word_vectorizer.joblib")
    mlb = joblib.load("saved_model/mlb.joblib")
    best_params = joblib.load("saved_model/best_params.joblib")

    print("Best params:")
    print(best_params)
    print()

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = load_combined_dataset()
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    y_test_bin = mlb.transform(y_test)

    print(f"Test samples: {len(X_test)}")
    print()

    print("=" * 80)
    print("TRANSFORMING TEST DATA")
    print("=" * 80)

    X_test_char = char_vectorizer.transform(X_test)
    X_test_word = word_vectorizer.transform(X_test)
    X_test_feat = hstack([X_test_char, X_test_word]).tocsr()

    print(f"Test feature shape: {X_test_feat.shape}")
    print()

    print("=" * 80)
    print("EVALUATION ON TEST DATA")
    print("=" * 80)

    y_pred_bin = model.predict(X_test_feat)

    subset_acc = accuracy_score(y_test_bin, y_pred_bin)
    micro_f1 = f1_score(y_test_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test_bin, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0)
    ham_loss = hamming_loss(y_test_bin, y_pred_bin)

    print(
        classification_report(
            y_test_bin,
            y_pred_bin,
            target_names=mlb.classes_,
            zero_division=0,
        )
    )

    print(f"accuracy:     {subset_acc:.4f}")
    print(f"macro avg f1: {macro_f1:.4f}")
    print(f"weighted f1:  {weighted_f1:.4f}")
    print(f"micro f1:     {micro_f1:.4f}")
    print(f"hamming loss: {ham_loss:.4f}")
    print()

    print("=" * 80)
    print("CONFUSION MATRIX PR. LABEL")
    print("=" * 80)

    mcm = multilabel_confusion_matrix(y_test_bin, y_pred_bin)

    for idx, label in enumerate(mlb.classes_):
        tn, fp, fn, tp = mcm[idx].ravel()
        print(f"{label}: TN={tn}  FP={fp}  FN={fn}  TP={tp}")


if __name__ == "__main__":
    main()