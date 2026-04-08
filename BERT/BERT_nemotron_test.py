# BERT_nemotron_test.py
# ─────────────────────────────────────────────
# Kører den trænede PII-model på observationer fra
# nvidia/Nemotron-PII datasættet fra Hugging Face.
# Sammenligner med ground truth hvis tilgængeligt.
#
# Kræver: pip install transformers torch datasets
# Model:  https://fromsmash.com/~M-eZgMNO1-dt
# ─────────────────────────────────────────────

import re
import ast
import time
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForTokenClassification

model_path    = "saved_model_combined"
max_len       = 512
device        = "cuda" if torch.cuda.is_available() else "cpu"
n_samples     = 1000
domain_filter = None   # F.eks. "Healthcare" – None = alle domæner

label_map = {
    "O": 0,
    "API_KEY": 1,
    "CREDIT_CARD_NUMBER": 2,
    "BANK_ACCOUNT_NUMBER": 3,
    "IBAN": 4,
    "PASSWORD": 5,
    "SSN": 6,
    "FULL_NAME": 7,
    "FIRST_NAME": 8,
    "LAST_NAME": 9,
    "EMAIL": 10,
    "PHONE_NUMBER": 11,
}

inv_label_map = {v: k for k, v in label_map.items()}

# Mapping fra Nemotron-labels til vores kategorier.
# Tom liste betyder labels ikke dækket af vores model – filtreres fra GT.
label_mapping = {
    "NAME":                   ["FULL_NAME", "FIRST_NAME", "LAST_NAME"],
    "FULL_NAME":              ["FULL_NAME"],
    "FIRST_NAME":             ["FIRST_NAME"],
    "LAST_NAME":              ["LAST_NAME"],
    "EMAIL":                  ["EMAIL"],
    "EMAIL_ADDRESS":          ["EMAIL"],
    "PHONE":                  ["PHONE_NUMBER"],
    "PHONE_NUMBER":           ["PHONE_NUMBER"],
    "SSN":                    ["SSN"],
    "SOCIAL_SECURITY_NUMBER": ["SSN"],
    "CREDIT_CARD":            ["CREDIT_CARD_NUMBER"],
    "CREDIT_CARD_NUMBER":     ["CREDIT_CARD_NUMBER"],
    "IBAN":                   ["IBAN"],
    "PASSWORD":               ["PASSWORD"],
    "API_KEY":                ["API_KEY"],
    "PASSPORT":               [],
    "PASSPORT_NUMBER":        [],
    "BANK_ACCOUNT":           ["BANK_ACCOUNT_NUMBER"],
    "BANK_ACCOUNT_NUMBER":    ["BANK_ACCOUNT_NUMBER"],
    "COMPANY_NAME":           [],
    "DATE_OF_BIRTH":          [],
    "DATE":                   [],
    "ADDRESS":                [],
    "STREET_ADDRESS":         [],
    "CITY":                   [],
    "ZIPCODE":                [],
    "USERNAME":               [],
    "RACE_ETHNICITY":         [],
    "GENDER":                 [],
    "RELIGION":               [],
}


class PIIDetector:

    def __init__(self, path: str = model_path):
        print(f"Indlæser model fra: {path}")
        self.device    = device
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model     = BertForTokenClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        print(f"Model klar – kører på: {self.device}\n")
        self.predict("warmup")

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"entities": [], "has_pii": False, "latency_ms": 0.0}

        start    = time.perf_counter()
        encoding = self.tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        probs       = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
        predictions = probs.argmax(dim=-1)
        tokens      = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        mask        = attention_mask[0].cpu()

        raw_tokens = []
        for idx, (token, pred, active, offset) in enumerate(
            zip(tokens, predictions, mask, offset_mapping)
        ):
            if not active or token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            raw_tokens.append({
                "token":      token,
                "label":      inv_label_map[pred.item()],
                "confidence": probs[idx][pred.item()].item(),
                "start":      offset[0].item(),
                "end":        offset[1].item(),
            })

        entities   = self._merge_entities(raw_tokens, text)
        entities   = self._fix_split_entities(entities, text)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "entities":   entities,
            "has_pii":    len(entities) > 0,
            "latency_ms": latency_ms,
        }

    def _fix_split_entities(self, entities: list, text: str) -> list:
        """
        Post-processing der fixer to kendte problemer:
        1. Emails splittet af tokenizeren – genfinder den fulde email via regex.
        2. Passwords der er startet midt i et ord – udvider baglæns til mellemrum.
        """
        email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
        fixed    = []
        used     = set()

        for i, ent in enumerate(entities):
            if i in used:
                continue

            if ent["label"] == "EMAIL":
                search_start = max(0, ent["start"] - 20)
                search_end   = min(len(text), ent["end"] + 60)
                match        = email_re.search(text[search_start:search_end])
                if match:
                    abs_start  = search_start + match.start()
                    abs_end    = search_start + match.end()
                    for j, other in enumerate(entities):
                        if j != i and other["label"] == "EMAIL":
                            if other["start"] >= abs_start and other["end"] <= abs_end + 5:
                                used.add(j)
                    fixed.append({
                        "text":       text[abs_start:abs_end],
                        "label":      "EMAIL",
                        "confidence": ent["confidence"],
                        "start":      abs_start,
                        "end":        abs_end,
                    })
                    continue

            if ent["label"] == "PASSWORD":
                word_start = ent["start"]
                while word_start > 0 and text[word_start - 1] not in (' ', '\t', '\n', ',', '('):
                    word_start -= 1
                word_end = ent["end"]
                while word_end < len(text) and text[word_end] not in (' ', '\t', '\n', ',', ')'):
                    word_end += 1
                fixed.append({
                    "text":       text[word_start:word_end],
                    "label":      "PASSWORD",
                    "confidence": ent["confidence"],
                    "start":      word_start,
                    "end":        word_end,
                })
                continue

            fixed.append(ent)

        return fixed

    def _merge_entities(self, raw_tokens: list, text: str) -> list:
        if not raw_tokens:
            return []

        entities = []
        current  = None

        for token in raw_tokens:
            label = token["label"]

            if token["token"].startswith("##"):
                if current and current["label"] == label:
                    current["end"]        = token["end"]
                    current["confidence"] = min(current["confidence"], token["confidence"])
                    current["text"]       = text[current["start"]:current["end"]]
                continue

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            if current and current["label"] != label:
                entities.append(current)
                current = None

            if current and current["label"] == label:
                gap = token["start"] - current["end"]
                if gap <= 2:
                    current["end"]        = token["end"]
                    current["confidence"] = min(current["confidence"], token["confidence"])
                    current["text"]       = text[current["start"]:current["end"]]
                else:
                    entities.append(current)
                    current = {
                        "text":       text[token["start"]:token["end"]],
                        "label":      label,
                        "confidence": token["confidence"],
                        "start":      token["start"],
                        "end":        token["end"],
                    }
            else:
                current = {
                    "text":       text[token["start"]:token["end"]],
                    "label":      label,
                    "confidence": token["confidence"],
                    "start":      token["start"],
                    "end":        token["end"],
                }

        if current:
            entities.append(current)

        return entities


def normalize_label(label: str) -> str:
    label = label.upper().replace(" ", "_")
    for key, our_labels in label_mapping.items():
        if label == key or label in our_labels:
            return key
    return label


def is_match(pred: dict, gt: dict) -> bool:
    if normalize_label(pred["label"]) != normalize_label(gt["label"]):
        return False
    p = pred["text"].lower().strip()
    g = gt["text"].lower().strip()
    return p in g or g in p


def parse_ground_truth(row: dict) -> list:
    """
    Udtrækker ground truth PII-entiteter fra spans-kolonnen.
    Filtrerer til kun labels vores model er trænet til at detektere,
    så precision/recall/F1 er fair sammenlignet med vores label_map.
    """
    gt  = []
    raw = row.get("spans", None)
    if not raw:
        return gt

    try:
        spans = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except Exception:
        return gt

    for span in spans:
        if not isinstance(span, dict):
            continue
        label = (span.get("label") or span.get("type") or span.get("entity_type") or "")
        text  = (span.get("text")  or span.get("value") or span.get("span") or "")
        if not label or not text:
            continue
        mapped = normalize_label(str(label).upper())
        if mapped in label_mapping and label_mapping[mapped]:
            gt.append({"text": str(text), "label": str(label).upper()})

    return gt


def load_nemotron_samples(n: int, domain: str = None) -> list:
    print("Indlæser nvidia/Nemotron-PII fra Hugging Face...")
    ds      = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
    samples = []

    for row in ds:
        if domain and row.get("domain", "") != domain:
            continue
        samples.append(row)
        if len(samples) >= n:
            break

    print(f"Indlæste {len(samples)} observationer\n")
    return samples


def run(samples: list, detector: PIIDetector):
    total_latency   = 0.0
    pii_count       = 0
    total_predicted = 0
    total_gt        = 0
    total_matched   = 0
    has_gt          = False

    for i, sample in enumerate(samples, start=1):
        text   = sample.get("text", "")
        result = detector.predict(text)
        gt     = parse_ground_truth(sample)

        if gt:
            has_gt = True

        total_latency   += result["latency_ms"]
        total_predicted += len(result["entities"])
        total_gt        += len(gt)

        if result["has_pii"]:
            pii_count += 1

        matched_gt = set()
        for pred in result["entities"]:
            for j, g in enumerate(gt):
                if j not in matched_gt and is_match(pred, g):
                    matched_gt.add(j)
                    total_matched += 1
                    break

        print(f"[{i}/{len(samples)}] {sample.get('domain','')} – "
              f"{sample.get('document_type','')}")
        print(f"  Tekst: {text[:120].replace(chr(10), ' ')}...")
        print(f"  Svartid: {result['latency_ms']:.1f} ms")

        if result["entities"]:
            print(f"  Fundet ({len(result['entities'])}):")
            for e in result["entities"]:
                matched = any(is_match(e, g) for g in gt)
                mark    = "✓" if matched else ("?" if not gt else "✗")
                print(f"    {mark}  {e['text']:<35} → {e['label']:<20} ({e['confidence']:.2f})")
        else:
            print(f"  Fundet: ingen")

        if gt:
            print(f"  Ground truth ({len(gt)}):")
            for g in gt:
                found = any(is_match(p, g) for p in result["entities"])
                print(f"    {'✓' if found else '✗'}  {g['text']:<35} → {g['label']}")
        else:
            print(f"  Ground truth: ikke tilgængeligt i dette dataset")

        print("─" * 65)

    avg_lat   = total_latency / len(samples) if samples else 0.0
    precision = total_matched / total_predicted if total_predicted > 0 else 0.0
    recall    = total_matched / total_gt        if total_gt > 0        else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'═' * 65}")
    print(f"  Observationer testet  : {len(samples)}")
    print(f"  Med PII (predicted)   : {pii_count} ({100 * pii_count / len(samples):.0f}%)")
    print(f"  Gns. svartid          : {avg_lat:.1f} ms")
    print(f"  NFR1 (≤100ms)         : {'✓ Opfyldt' if avg_lat <= 100 else '✗ Ikke opfyldt'}")

    if has_gt:
        print(f"\n  Ground truth sammenligning (kun labels i vores model):")
        print(f"  PII entiteter predicted : {total_predicted}")
        print(f"  PII entiteter i GT      : {total_gt}")
        print(f"  Korrekt matchede        : {total_matched}")
        print(f"  Precision               : {precision:.2f}")
        print(f"  Recall                  : {recall:.2f}")
        print(f"  F1                      : {f1:.2f}")
    else:
        print(f"\n  Ground truth ikke tilgængeligt.")
        print(f"  PII entiteter predicted i alt: {total_predicted}")

    print(f"{'═' * 65}")


if __name__ == "__main__":
    detector = PIIDetector()
    samples  = load_nemotron_samples(n_samples, domain=domain_filter)
    run(samples, detector)