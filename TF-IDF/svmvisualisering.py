import os
import sys
import ast
import joblib
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_combined_dataset


TEXT_COLUMN = "source_text"
PRIVACY_COLUMN = "privacy"
RANDOM_STATE = 42
LABEL_TO_VISUALIZE = "EMAIL"
N_SAMPLES = 5000

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


def hinge_loss(margin):
    return np.maximum(0, 1 - margin)


def modified_huber_loss(margin):
    return np.where(
        margin >= 1,
        0,
        np.where(
            margin >= -1,
            (1 - margin) ** 2,
            -4 * margin,
        ),
    )


def main():
    print("Loading saved model and vectorizers...")

    model = joblib.load("saved_model/model.joblib")
    char_vectorizer = joblib.load("saved_model/char_vectorizer.joblib")
    word_vectorizer = joblib.load("saved_model/word_vectorizer.joblib")
    mlb = joblib.load("saved_model/mlb.joblib")

    print("Loading dataset...")
    df = load_combined_dataset()

    df = df[[TEXT_COLUMN, PRIVACY_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df["labels"] = df[PRIVACY_COLUMN].apply(extract_labels_from_privacy)
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    X_text = df[TEXT_COLUMN]
    y = df["labels"].apply(
        lambda labels: 1 if LABEL_TO_VISUALIZE in labels else 0
    ).values

    X_small, _, y_small, _ = train_test_split(
        X_text,
        y,
        train_size=N_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Samples used: {len(X_small)}")
    print(f"{LABEL_TO_VISUALIZE} samples: {sum(y_small)}")
    print(f"NOT {LABEL_TO_VISUALIZE} samples: {len(y_small) - sum(y_small)}")

    print("Transforming text with saved TF-IDF vectorizers...")
    X_char = char_vectorizer.transform(X_small)
    X_word = word_vectorizer.transform(X_small)
    X_feat = hstack([X_char, X_word]).tocsr()

    label_index = list(mlb.classes_).index(LABEL_TO_VISUALIZE)
    email_classifier = model.estimators_[label_index]

    print("Calculating decision scores...")
    scores = email_classifier.decision_function(X_feat)

    y_signed = np.where(y_small == 1, 1, -1)
    margin = y_signed * scores

    hinge = hinge_loss(margin)
    huber = modified_huber_loss(margin)

    email_mask = y_small == 1
    not_email_mask = y_small == 0

    print("Creating plot...")

    plt.figure(figsize=(11, 7))

    plt.scatter(
        scores[email_mask],
        hinge[email_mask],
        alpha=0.45,
        s=18,
        label="Hinge loss - EMAIL",
        color="royalblue",
    )

    plt.scatter(
        scores[not_email_mask],
        hinge[not_email_mask],
        alpha=0.45,
        s=18,
        label="Hinge loss - NOT EMAIL",
        color="crimson",
    )

    plt.scatter(
        scores[email_mask],
        huber[email_mask],
        alpha=0.45,
        s=18,
        label="Modified Huber loss - EMAIL",
        color="seagreen",
    )

    plt.scatter(
        scores[not_email_mask],
        huber[not_email_mask],
        alpha=0.45,
        s=18,
        label="Modified Huber loss - NOT EMAIL",
        color="darkorange",
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Decision boundary",
    )

    plt.xlabel("SVM decision score")
    plt.ylabel("Loss")
    plt.title("Hinge vs Modified Huber loss on real EMAIL predictions")

    plt.legend()
    plt.tight_layout()

    output_file = "hinge_vs_modified_huber_email_4colors.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure as: {output_file}")


if __name__ == "__main__":
    main()