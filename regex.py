""" This script applies predefined regex patterns to identify structured PII types.
    The model is deterministic and does not require training."""

import re
from data_loader import load_combined_dataset, LABEL_MAP

class RegexModel:
    """
    A rule-based model for detecting Personally Identifiable 
    Information (PII) by using regular expressions.
    """

    def __init__(self):
        """
        Initialize regex patterns for selected PII labels. Only labels 
        that can be reliably detected via structure are included.
        """
        self.patterns = {
            "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}"),
            "PHONE_NUMBER": re.compile(r"\+?\d[\d\s-]{6,}\d"),
            "CREDIT_CARD_NUMBER": re.compile(r"(?:\d{4}[ -]?){3}\d{4}"),
            "SSN": re.compile(r"\d{6}-\d{4}")
        }

    def predict(self, text):
        """
        Detect PII entities in a given text.

        Parameters
        ----------
        text : str
            Input text to analyze.

        Returns
        -------
        list of dict
            List of detected entities:
            [
                {"label": str, "value": str},
                ...
            ]
        """
        predictions = []

        # Loop through each regex pattern
        for label, pattern in self.patterns.items():

            # Find all matches in the text
            for match in pattern.finditer(text):
                predictions.append({
                    "label": label,
                    "value": match.group()  # Extract matched substring
                })

        return predictions


def evaluate_model(df, model):
    """
    Evaluate the regex model using precision and recall.
    """

    # Counters for evaluation
    tp = 0  
    fp = 0  
    fn = 0 

    # Loop through entire dataset
    for i, (_, row) in enumerate(df.iterrows()):
        text = row["source_text"]
        true_labels = row["privacy"]
    
        # Get predictions from model
        preds = model.predict(text)

        # Convert to sets for comparison
        true_set = {(e["label"], e["value"]) for e in true_labels}
        pred_set = {(p["label"], p["value"]) for p in preds}

        # Update counters
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

        # Optional: print a few examples for debugging
        if i < 5 and preds:
            print("TEXT:", text)
            print("PREDICTIONS:", preds)
            print("GROUND TRUTH:", true_labels)
            print("-" * 50)

    # Compute metrics
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    
    # 1e-9 is a very small value added to avoid division by zero when there are no true 
    # or predicted positives. It ensures numerical stability without affecting the result.

    print("\n--- Evaluation Results ---")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}") # Just show 4 decimal places
    print(f"Recall:          {recall:.4f}")


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Loads dataset, initializes model, and runs evaluation.
    """

    # Load dataset 
    df = load_combined_dataset()

    model = RegexModel()
    
    evaluate_model(df, model)