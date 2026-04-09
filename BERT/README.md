# BERT PII Detection

This folder contains the BERT-based PII detection pipeline developed for the project.

## Setup

### Requirements
Python 3.10+ and the following packages:

```bash
pip install transformers torch tqdm pandas scikit-learn pypdf datasets
```

> **Note:** If running on AAU AI Lab, packages are already available in the Singularity container. See the run command below.

### Download the trained model
The model files are too large for GitHub and must be downloaded separately.

**Download `saved_model_combined` (recommended) from:**
> https://fromsmash.com/~M-eZgMNO1-dt

Unzip and place the model folder in the `BERT/` directory so the structure looks like:
```
BERT/
├── saved_model_combined/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── BERT_inference.py
└── ...
```

## Running inference

```bash
python BERT_inference.py
```

This will load the trained model and run PII detection on `pii_test_document.pdf`. No training required.

## Running external dataset test

```bash
python BERT_nemotron_test.py
```

Runs the model on observations from `nvidia/Nemotron-PII` from Hugging Face and compares predictions against ground truth. Adjust `n_samples` and `domain_filter` at the top of the file.

## Running on AAU AI Lab

```bash
srun --time=06:00:00 singularity exec --nv \
  -B ~/BERT/my_venv:/scratch/my_venv \
  -B $HOME/.singularity:/scratch/singularity \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -c "source /scratch/my_venv/bin/activate && python ~/BERT/BERT_imp_v6.py"
```

---

## Changelog

### BERT_imp_v1
`BertForSequenceClassification` implemented on the dataset. This model does not classify individual tokens but performs binary classification over each text span to determine whether it contains PII or not.

### BERT_imp_v2
Switched to `BertForTokenClassification` which classifies each token individually. Tested with 5 labels over 3 epochs, achieving F1 ~0.90. However, due to severe class imbalance in the dataset, expanding to more labels degraded performance significantly.

### BERT_imp_v3
Model optimised with class weights — the `O` (non-PII) class weighted at 0.3 to counter the heavy imbalance. Full dataset used for training/validation/test. Extended to 25 labels including financial, identity and personal info categories. Macro avg F1: **0.89**.

### BERT_imp_v4
Reduced label map from 25 to 13 labels based on priority input from the project partner. The hypothesis was that fewer, well-supported labels would improve F1 by removing classes with too little training data (e.g. `ROUTING_NUMBER`, `COORDINATES`). Labels kept:

| Category | Labels |
|---|---|
| Financial/access | `API_KEY`, `CREDIT_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `IBAN` |
| Identity/access | `PASSWORD`, `PASSPORT_NUMBER`, `SSN` |
| Personal info | `FULL_NAME`, `FIRST_NAME`, `LAST_NAME`, `EMAIL`, `PHONE_NUMBER` |

Early stopping added to automatically save the best model epoch. Macro avg F1: **0.89**, Weighted avg F1: **0.99**.

### BERT_imp_v5
Structured data filtering added to the preprocessing pipeline before training. Texts resembling JSON, CSV, or log-format data are removed using regex detection and special character density (>5% threshold). The hypothesis was that the model was learning to recognise PII via structural markers (colons, quotes) rather than linguistic context.

Results confirmed the hypothesis — macro avg F1 improved from **0.89 → 0.91** despite fewer training examples after filtering. `PASSPORT_NUMBER` improved significantly from 0.67 → 0.82 and `IBAN` from 0.87 → 0.97. External test on `nvidia/Nemotron-PII` (1000 observations) achieved F1 **0.64** at **22.9ms average latency**, satisfying NFR1. This is the final production model (`saved_model_v5`).

### BERT_imp_v6
Training data expanded by combining two datasets: `syvai/pii-dataset-eng` and `nvidia/Nemotron-PII` (466,150 rows combined, 428,697 after filtering + augmentation). Key changes:

- `PASSPORT_NUMBER` removed from label map (poor F1 due to Nemotron not annotating passport numbers in text)
- Credit card data: less aggressive structured-data filtering to retain credit card examples, plus format augmentation (+31,323 rows)
- AMP (Automatic Mixed Precision) now correctly uses `autocast` and `GradScaler` throughout the training loop
- Updated to non-deprecated `torch.amp` API
- Device selection is now automatic: falls back to CPU if no CUDA GPU is detected

**Results** (12 labels, early stop epoch 3/10, 455 min 57 sek on GPU):

| Label | Precision | Recall | F1 |
|---|---|---|---|
| O | 1.00 | 1.00 | 1.00 |
| API_KEY | 0.99 | 0.99 | 0.99 |
| CREDIT_CARD_NUMBER | 0.97 | 1.00 | 0.98 |
| BANK_ACCOUNT_NUMBER | 0.91 | 0.85 | 0.88 |
| IBAN | 0.94 | 0.94 | 0.94 |
| PASSWORD | 0.98 | 0.99 | 0.98 |
| SSN | 0.90 | 0.97 | 0.94 |
| FULL_NAME | 0.74 | 0.80 | 0.77 |
| FIRST_NAME | 0.89 | 0.93 | 0.91 |
| LAST_NAME | 0.87 | 0.91 | 0.89 |
| EMAIL | 0.99 | 1.00 | 1.00 |
| PHONE_NUMBER | 0.97 | 0.99 | 0.98 |

Macro avg F1: **0.94**, Weighted avg F1: **1.00**, Accuracy: **1.00**

### BERT_inference
Inference-only script that loads a saved model and runs PII detection on a PDF without retraining. Includes:
- Latency measurement per request (~6–10ms on GPU, well within the 100ms NFR requirement)
- Confidence scores per detected entity
- PDF loading and sentence splitting via `pypdf`
- Post-processing to fix tokenizer artifacts (split emails, truncated passwords)

### BERT_nemotron_test
External evaluation script that loads observations from `nvidia/Nemotron-PII` and compares model predictions against ground truth. Ground truth is filtered to only the labels covered by our model for a fair comparison. Reports precision, recall, F1 and average latency across all tested observations.
