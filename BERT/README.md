# BERT PII Detection

This folder contains the BERT-based PII detection pipeline developed for the project.

## Setup

### Requirements
Python 3.10+ and the following packages:

```bash
pip install transformers torch tqdm pandas scikit-learn pypdf
```

> **Note:** If running on AAU AI Lab, packages are already available in the Singularity container. See the run command below.

### Download the trained model
The model files are too large for GitHub and must be downloaded separately.

**Download `saved_model_reduced` from:**
> https://www.dropbox.com/t/yhU1bqLll8bMo166

Unzip and place the `saved_model_reduced` folder in the `BERT/` directory so the structure looks like:
```
BERT/
├── saved_model_reduced/
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

## Running on AAU AI Lab

```bash
srun --time=06:00:00 singularity exec --nv \
  -B ~/BERT/my_venv:/scratch/my_venv \
  -B $HOME/.singularity:/scratch/singularity \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -c "source /scratch/my_venv/bin/activate && python ~/BERT/BERT_imp_v4.py"
```

---

## Changelog

### BERT_imp_v1
`BertForSequenceClassification` implemented on the dataset. This model does not classify individual tokens but performs binary classification over each text span to determine whether it contains PII or not.

### BERT_imp_v2
Switched to `BertForTokenClassification` which classifies each token individually. Tested with 5 labels over 3 epochs, achieving F1 ~0.90. However, due to severe class imbalance in the dataset, expanding to more labels degraded performance significantly.

### BERT_imp_v3
Model optimised with class weights — the `O` (non-PII) class weighted at 0.1 to counter the heavy imbalance. Full dataset used for training/validation/test. Extended to 25 labels including financial, identity and personal info categories. Macro avg F1: **0.87–0.89**.

### BERT_imp_v4
Reduced label map from 25 to 13 labels based on priority input from the project partner (Danni). The hypothesis was that fewer, well-supported labels would improve F1 by removing classes with too little training data (e.g. `ROUTING_NUMBER`, `COORDINATES`). Labels kept:

| Category | Labels |
|---|---|
| Financial/access | `API_KEY`, `CREDIT_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `IBAN` |
| Identity/access | `PASSWORD`, `PASSPORT_NUMBER`, `SSN` |
| Personal info | `FULL_NAME`, `FIRST_NAME`, `LAST_NAME`, `EMAIL`, `PHONE_NUMBER` |

Early stopping added to automatically save the best model epoch. Macro avg F1: **0.89**, Weighted avg F1: **0.99**.

### BERT_inference
Inference-only script that loads `saved_model_reduced` and runs PII detection on a PDF without retraining. Includes:
- Latency measurement per request (~6–10ms, well within the 100ms NFR requirement)
- Confidence scores per detected entity
- PDF loading and sentence splitting via `pypdf`
- Post-processing to fix tokenizer artifacts (split emails, truncated passwords)