# BERT Model Results

Comparison of training results across model versions. All versions use `bert-base-uncased` fine-tuned on the [syvai/pii-dataset-eng](https://huggingface.co/datasets/syvai/pii-dataset-eng) dataset.

---

## Overview

| Version | Dataset size | Epochs | Labels | Training time | Macro F1 | Weighted F1 | Accuracy |
|---------|-------------|--------|--------|---------------|----------|-------------|----------|
| V2      | 250,000     | 3      | 25     | 83 min 22 sek | 0.82     | 0.98        | 0.98     |
| V3      | 266,000     | 4 (early stop at 5) | 25 | 194 min 34 sek | 0.89 | 0.98 | 0.98 |
| V4      | 266,000     | 3 (early stop at 5) | 13 | 125 min 25 sek | 0.89 | 0.99 | 0.99 |
| V5      | 266,000 (filtered) | 3 (early stop at 5) | 13 | 121 min 30 sek | 0.91 | 0.99 | 0.99 |

---

## V2 — Baseline (25 labels, 3 epochs, 250k samples)

**Training:**
```
Træning:    175000 eksempler
Validering: 37500 eksempler
Test:       37500 eksempler

Epoch 1/3  |  Train loss: 0.2145  |  Val loss: 0.0330  |  Val accuracy: 0.9815
Epoch 2/3  |  Train loss: 0.0290  |  Val loss: 0.0280  |  Val accuracy: 0.9838
Epoch 3/3  |  Train loss: 0.0230  |  Val loss: 0.0261  |  Val accuracy: 0.9849

Træning færdig – tog 83 min 22 sek
```

**Evaluation:**
```
                       precision    recall  f1-score   support

                    O       0.99      0.99      0.99   2402595
              API_KEY       0.91      0.94      0.92      1109
   CREDIT_CARD_NUMBER       0.89      0.97      0.93     14136
      CREDIT_CARD_CVV       0.00      0.00      0.00        67
  BANK_ACCOUNT_NUMBER       0.94      0.91      0.93      1083
       ROUTING_NUMBER       0.59      0.70      0.64       446
                 IBAN       0.88      0.96      0.92      1205
             PASSWORD       0.97      0.98      0.98     11419
      PASSPORT_NUMBER       0.80      0.66      0.72       479
                  SSN       0.95      0.92      0.94     15587
DRIVER_LICENSE_NUMBER       0.94      0.95      0.94     11560
           TAX_NUMBER       0.94      0.92      0.93     10729
            FULL_NAME       0.89      0.83      0.86     24100
                EMAIL       0.99      0.99      0.99     45320
         PHONE_NUMBER       0.98      0.99      0.99     31168
        DATE_OF_BIRTH       0.88      0.83      0.86     13153
       STREET_ADDRESS       0.88      0.95      0.91     27790
                 CITY       0.89      0.93      0.91     17432
              ZIPCODE       0.90      0.95      0.92      5480
                 DATE       0.85      0.90      0.87     18109
             USERNAME       0.97      0.93      0.95     28711
              COMPANY       0.79      0.83      0.81     15260
              IPV4_ID       0.00      0.00      0.00         0
                 IPV6       0.98      0.99      0.99      1762
          COORDINATES       0.65      0.78      0.71       653

             accuracy                           0.98   2699353
            macro avg       0.82      0.83      0.82   2699353
         weighted avg       0.99      0.98      0.98   2699353
```

**Notes:**
- `CREDIT_CARD_CVV` and `IPV4_ID` scored F1 0.00 due to insufficient training data (67 and 0 examples respectively)
- `ROUTING_NUMBER` (0.64) and `COORDINATES` (0.71) weakest performers due to low support

---

## V3 — Full dataset + early stopping + class weights (25 labels, up to 5 epochs, 266k samples)

**Training:**
```
Træning:    186200 eksempler
Validering: 39900 eksempler
Test:       39900 eksempler

Epoch 1/5  |  Train loss: 0.3383  |  Val loss: 0.0794  |  Val accuracy: 0.9703  ✓ saved
Epoch 2/5  |  Train loss: 0.0690  |  Val loss: 0.0633  |  Val accuracy: 0.9777  ✓ saved
Epoch 3/5  |  Train loss: 0.0522  |  Val loss: 0.0599  |  Val accuracy: 0.9772  ✓ saved
Epoch 4/5  |  Train loss: 0.0427  |  Val loss: 0.0597  |  Val accuracy: 0.9801  ✓ saved
Epoch 5/5  |  Train loss: 0.0367  |  Val loss: 0.0598  |  Val accuracy: 0.9806  ✗ early stop

Best model: epoch 4 (val loss: 0.0597)
Træning færdig – tog 194 min 34 sek
```

**Evaluation:**
```
                       precision    recall  f1-score   support

                    O       1.00      0.98      0.99   2528121
              API_KEY       0.94      0.92      0.93      1443
   CREDIT_CARD_NUMBER       0.88      0.99      0.93     14737
  BANK_ACCOUNT_NUMBER       0.89      0.93      0.91      1090
       ROUTING_NUMBER       0.52      0.82      0.64       471
                 IBAN       0.87      0.98      0.92      1253
             PASSWORD       0.97      0.99      0.98     11786
      PASSPORT_NUMBER       0.76      0.81      0.79       502
                  SSN       0.95      0.94      0.94     16614
DRIVER_LICENSE_NUMBER       0.93      0.97      0.95     12662
           TAX_NUMBER       0.92      0.94      0.93     11325
            FULL_NAME       0.77      0.89      0.83     25958
           FIRST_NAME       0.82      0.89      0.85     19864
            LAST_NAME       0.76      0.80      0.78     16242
                EMAIL       0.99      0.99      0.99     48563
         PHONE_NUMBER       0.98      1.00      0.99     32957
        DATE_OF_BIRTH       0.78      0.92      0.84     13997
       STREET_ADDRESS       0.84      0.98      0.90     29960
                 CITY       0.89      0.96      0.92     18353
              ZIPCODE       0.86      0.98      0.91      5943
                 DATE       0.79      0.95      0.86     19670
             USERNAME       0.98      0.94      0.96     29532
              COMPANY       0.69      0.91      0.78     17230
                 IPV6       0.94      0.99      0.96      1532
          COORDINATES       0.55      0.97      0.70       694

             accuracy                           0.98   2880499
            macro avg       0.85      0.94      0.89   2880499
         weighted avg       0.98      0.98      0.98   2880499
```

**Notes:**
- Macro F1 improved significantly from 0.82 → 0.89 due to class weights and more data
- Early stopping correctly identified epoch 4 as optimal
- `ROUTING_NUMBER` and `COORDINATES` still weak due to low support

---

## V4 — Reduced label map (13 labels, up to 5 epochs, 266k samples)

**Training:**
```
Træning:    186200 eksempler
Validering: 39900 eksempler
Test:       39900 eksempler

Early stopping triggered after epoch 3 (val loss: 0.0334)
Best model: epoch 3

Træning færdig – tog 125 min 25 sek
```

**Evaluation:**
```
                     precision    recall  f1-score   support

                  O       1.00      0.99      1.00   2689320
            API_KEY       0.91      0.97      0.94      1443
 CREDIT_CARD_NUMBER       0.85      0.99      0.92     14737
BANK_ACCOUNT_NUMBER       0.93      0.92      0.93      1090
               IBAN       0.79      0.98      0.87      1253
           PASSWORD       0.97      0.99      0.98     11786
    PASSPORT_NUMBER       0.55      0.86      0.67       502
                SSN       0.89      0.97      0.92     16638
          FULL_NAME       0.74      0.89      0.81     25987
         FIRST_NAME       0.82      0.87      0.84     19864
          LAST_NAME       0.72      0.82      0.77     16242
              EMAIL       0.99      0.99      0.99     48680
       PHONE_NUMBER       0.98      1.00      0.99     32957

           accuracy                           0.99   2880499
          macro avg       0.86      0.94      0.89   2880499
       weighted avg       0.99      0.99      0.99   2880499
```

**Notes:**
- Macro F1 unchanged at 0.89 despite reducing labels — `PASSPORT_NUMBER` dropped from 0.79 → 0.67
- Weighted F1 improved slightly to 0.99 due to removal of weak labels
- Training time reduced by ~35% (194 → 125 min) with fewer labels
- This is the model used in `BERT_inference.py` (`saved_model_reduced`)

---

## V5 — Structured data filtering (13 labels, up to 5 epochs, 266k samples filtered)

**Changes from V4:**
- Structured data (JSON, CSV, log-format tekster) filtreret fra træningsdata via regex-detektion og tegndensitet (>5% specialtegn)
- Hypotesen: modellen lærer kontekstuel PII-genkendelse fremfor at lære strukturelle mønstre som kolon og anførselstegn

**Training:**
```
Træning:    ~163000 eksempler (efter filtrering)
Validering: ~35000 eksempler
Test:       ~35000 eksempler

Early stopping triggered after epoch 3
Best model: epoch 3

Træning færdig – tog 121 min 30 sek
```

**Evaluation:**
```
                     precision    recall  f1-score   support

                  O       1.00      0.99      1.00   2019385
            API_KEY       0.89      0.87      0.88       606
 CREDIT_CARD_NUMBER       0.89      0.97      0.92     12215
BANK_ACCOUNT_NUMBER       0.88      0.97      0.93       506
               IBAN       0.95      1.00      0.97       634
           PASSWORD       0.96      0.99      0.97      9329
    PASSPORT_NUMBER       0.75      0.91      0.82       248
                SSN       0.90      0.97      0.93     14137
          FULL_NAME       0.74      0.88      0.81     12459
         FIRST_NAME       0.82      0.88      0.85     17110
          LAST_NAME       0.76      0.82      0.79     13583
              EMAIL       1.00      1.00      1.00     41593
       PHONE_NUMBER       0.99      1.00      0.99     27305

           accuracy                           0.99   2169110
          macro avg       0.89      0.94      0.91   2169110
       weighted avg       0.99      0.99      0.99   2169110
```

**Notes:**
- Macro F1 improved from 0.89 → 0.91 trods færre træningseksempler efter filtrering
- `PASSPORT_NUMBER` forbedret markant fra 0.67 (V4) → 0.82 — sandsynligvis fordi struktureret støj forvirrede modellen på sjældne labels
- `IBAN` forbedret fra 0.87 → 0.97
- `API_KEY` svagt fald fra 0.94 → 0.88, muligvis fordi API keys ofte optræder i struktureret kontekst der nu er filtreret fra
- This is the final model used in production (`saved_model_v5`)

---

## Ekstern test — nvidia/Nemotron-PII (1000 observationer)

Modellerne er testet på [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) som et uafhængigt datasæt fra et andet domæne. Ground truth er filtreret til kun de labels vores model dækker for fair sammenligning.

| Version | max_len | Precision | Recall | F1   | Gns. svartid | NFR1 (≤100ms) |
|---------|---------|-----------|--------|------|--------------|---------------|
| V4      | 128     | 0.79      | 0.45   | 0.57 | 102 ms       | ✗             |
| V4      | 256     | –         | –      | 0.61 | –            | –             |
| V4      | 512     | 0.72      | 0.58   | 0.64 | 174 ms       | ✗             |
| V5      | 512     | 0.72      | 0.58   | 0.64 | **22.9 ms**  | ✓             |

**Notes:**
- V5 opnår samme F1 (0.64) som V4 med max_len=512, men ved max_len=128 og 7x lavere svartid
- Svartidsforbedringen fra 174ms → 22.9ms skyldes primært GPU-inference på AAU AI Lab vs. CPU lokalt
- Recall-loftet på 0.58 er delvist strukturelt: Nemotron indeholder labels som `MEDICAL_RECORD_NUMBER`, `VEHICLE_IDENTIFIER`, `MAC_ADDRESS` der aldrig er i vores GT-filter men stadig udgør en del af teksterne
- Domænegabet mellem syvai (syntetisk, struktureret) og Nemotron (realistisk, varieret) forklarer gabet mellem intern F1 (0.91) og ekstern F1 (0.64)

---

## Label performance across versions

Labels present in all versions:

| Label              | V2 F1 | V3 F1 | V4 F1 | V5 F1 |
|--------------------|-------|-------|-------|-------|
| API_KEY            | 0.92  | 0.93  | 0.94  | 0.88  |
| CREDIT_CARD_NUMBER | 0.93  | 0.93  | 0.92  | 0.92  |
| BANK_ACCOUNT_NUMBER| 0.93  | 0.91  | 0.93  | 0.93  |
| IBAN               | 0.92  | 0.92  | 0.87  | 0.97  |
| PASSWORD           | 0.98  | 0.98  | 0.98  | 0.97  |
| PASSPORT_NUMBER    | 0.72  | 0.79  | 0.67  | 0.82  |
| SSN                | 0.94  | 0.94  | 0.92  | 0.93  |
| FULL_NAME          | 0.86  | 0.83  | 0.81  | 0.81  |
| EMAIL              | 0.99  | 0.99  | 0.99  | 1.00  |
| PHONE_NUMBER       | 0.99  | 0.99  | 0.99  | 0.99  |
| FIRST_NAME         | –     | 0.85  | 0.84  | 0.85  |
| LAST_NAME          | –     | 0.78  | 0.77  | 0.79  |