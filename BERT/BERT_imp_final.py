# BERT_imp_final.py
# ─────────────────────────────────────────────
# Deployment-klar PII-detektion med den trænede BERT-model.
# Indlæser saved_model_combined og eksponerer PIIDetector til GUI-brug.
#
# Brug fra GUI:
#   from BERT_imp_final import PIIDetector
#   detector = PIIDetector()
#   result = detector.predict("Ring til Jonas Hansen på Jonas@gmail.com")
#   print(result.entities)   # [{"text": "Jonas Hansen", "label": "FULL_NAME", ...}, ...]
#   print(result.has_pii)    # True
#
# Model kan downloades her: https://fromsmash.com/~M-eZgMNO1-dt
# ─────────────────────────────────────────────

import re
import os
import time
import torch
from transformers import BertTokenizer, BertForTokenClassification

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model_combined")
MAX_LEN    = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
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

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class PIIResult:
    """
    Resultat fra PIIDetector.predict().

    Attributter:
        text           : Den originale inputtekst
        entities       : Liste af fundne PII-entiteter –
                         [{"text": ..., "label": ..., "confidence": ..., "start": ..., "end": ...}, ...]
        has_pii        : True hvis der blev fundet PII
        latency_ms     : Målt svartid i millisekunder
        confidence_avg : Gennemsnitlig confidence på tværs af fundne entiteter
    """

    def __init__(self, text: str, entities: list, latency_ms: float):
        self.text           = text
        self.entities       = entities
        self.has_pii        = len(entities) > 0
        self.latency_ms     = latency_ms
        self.confidence_avg = (
            sum(e["confidence"] for e in entities) / len(entities)
            if entities else 0.0
        )

    def __repr__(self):
        status = "PII FUNDET" if self.has_pii else "Ingen PII"
        lines  = [
            f"[{status}] '{self.text}'",
            f"  Svartid: {self.latency_ms:.1f} ms  |  Gns. confidence: {self.confidence_avg:.2f}",
        ]
        if self.has_pii:
            lines.append("  " + "─" * 55)
            for e in self.entities:
                lines.append(
                    f"  {e['text']:<30} → {e['label']:<25} ({e['confidence']:.2f})"
                )
        return "\n".join(lines)


class PIIDetector:
    """
    Indlæser saved_model_combined og kører PII-detektion på tekst.
    Singleton-venlig: opret én instans og genbrug den på tværs af GUI-kald.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"Indlæser model fra: {model_path}")
        self.device    = DEVICE
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model     = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model klar – kører på: {self.device}")
        self._warmup()

    def _warmup(self):
        self.predict("warmup")

    def predict(self, text: str) -> PIIResult:
        """
        Analyserer en tekst og returnerer et PIIResult objekt.

        Args:
            text: Den tekst der skal analyseres (fra GUI-input)

        Returns:
            PIIResult med entities, has_pii, confidence scores og latency
        """
        if not text or not text.strip():
            return PIIResult(text=text, entities=[], latency_ms=0.0)

        start_time = time.perf_counter()

        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
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
                "label":      INV_LABEL_MAP[pred.item()],
                "confidence": probs[idx][pred.item()].item(),
                "start":      offset[0].item(),
                "end":        offset[1].item(),
            })

        entities   = self._merge_entities(raw_tokens, text)
        entities   = self._fix_split_entities(entities, text)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return PIIResult(text=text, entities=entities, latency_ms=latency_ms)

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

    def _fix_split_entities(self, entities: list, text: str) -> list:
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
                    abs_start = search_start + match.start()
                    abs_end   = search_start + match.end()
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
