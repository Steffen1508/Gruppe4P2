import re
import torch
import time
from transformers import BertTokenizer, BertForTokenClassification

MODEL_PATH = "saved_model_combined"
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Labels – skal matche label_map fra træningsscriptet (13 labels, 0-12)
label_map = {
    "O": 0,
    # Finansielt/adgang
    "API_KEY": 1,
    "CREDIT_CARD_NUMBER": 2,
    "BANK_ACCOUNT_NUMBER": 3,
    "IBAN": 4,
    # Identitet/adgang
    "PASSWORD": 5,
    "PASSPORT_NUMBER": 6,
    "SSN": 7,
    # Personlig info
    "FULL_NAME": 8,
    "FIRST_NAME": 9,
    "LAST_NAME": 10,
    "EMAIL": 11,
    "PHONE_NUMBER": 12,
}

inv_label_map = {v: k for k, v in label_map.items()}


class PIIDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        print(f"Indlæser model fra: {model_path}")
        self.device = DEVICE
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model klar – kører på: {self.device}")
        # Warm-up så første rigtige forespørgsel ikke rammes af cold start
        self.predict("warmup")

    def predict(self, text: str) -> "PIIResult":
        if not text or not text.strip():
            raise ValueError("Tom tekst – indtast venligst noget tekst.")

        start_time = time.perf_counter()

        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Softmax for at få confidence scores per token
        probs = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
        predictions = probs.argmax(dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        mask = attention_mask[0].cpu()

        raw_tokens = []
        for idx, (token, pred, active, offset) in enumerate(zip(tokens, predictions, mask, offset_mapping)):
            if not active or token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            label = inv_label_map[pred.item()]
            confidence = probs[idx][pred.item()].item()
            raw_tokens.append({
                "token":      token,
                "label":      label,
                "confidence": confidence,
                "start":      offset[0].item(),
                "end":        offset[1].item(),
            })

        # Slå WordPiece-tokens og sammenhængende entiteter sammen
        entities = self._merge_entities(raw_tokens, text)
        # Fix emails og passwords der er splittet af modellen
        entities = self._fix_split_entities(entities, text)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return PIIResult(text=text, entities=entities, latency_ms=latency_ms)

    def _fix_split_entities(self, entities: list, text: str) -> list:
        # Finder den fulde email i original tekst via regex i stedet for at
        # stole på model-output, da modellen tit splitter emails forkert
        email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

        fixed = []
        used = set()

        for i, ent in enumerate(entities):
            if i in used:
                continue

            if ent["label"] == "EMAIL":
                search_start = max(0, ent["start"] - 20)
                search_end = min(len(text), ent["end"] + 60)
                snippet = text[search_start:search_end]
                match = email_re.search(snippet)
                if match:
                    abs_start = search_start + match.start()
                    abs_end = search_start + match.end()
                    # Marker overlappende EMAIL-fragmenter som brugte
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
                # Udvid baglæns og fremad til ordgrænse, da modellen
                # tit kun fanger en del af passwordet
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
        current = None

        for token in raw_tokens:
            label = token["label"]

            # WordPiece ## tokens hører altid til forrige token
            if token["token"].startswith("##"):
                if current and current["label"] == label:
                    current["end"] = token["end"]
                    current["confidence"] = min(current["confidence"], token["confidence"])
                    # Rekonstruer fra original tekst så specialtegn bevares
                    current["text"] = text[current["start"]:current["end"]]
                continue

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            # Ny label – gem nuværende entitet og start forfra
            if current and current["label"] != label:
                entities.append(current)
                current = None

            if current and current["label"] == label:
                # Udvid kun hvis tokens er tæt nok (maks 2 tegn mellemrum)
                # — fanger f.eks. "kevin.larsen @ corp.net" men ikke adskilte entiteter
                gap = token["start"] - current["end"]
                if gap <= 2:
                    current["end"] = token["end"]
                    current["confidence"] = min(current["confidence"], token["confidence"])
                    current["text"] = text[current["start"]:current["end"]]
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


class PIIResult:
    def __init__(self, text: str, entities: list, latency_ms: float):
        self.text = text
        self.entities = entities
        self.has_pii = len(entities) > 0
        self.latency_ms = latency_ms
        self.confidence_avg = (
            sum(e["confidence"] for e in entities) / len(entities)
            if entities else 0.0
        )

    def __repr__(self):
        status = "✓ PII FUNDET" if self.has_pii else "✗ Ingen PII"
        lines = [
            f"[{status}] '{self.text}'",
            f"  Svartid: {self.latency_ms:.1f} ms  |  Gns. confidence: {self.confidence_avg:.2f}",
        ]
        if self.has_pii:
            lines.append("  " + "─" * 55)
            for e in self.entities:
                lines.append(f"  {e['text']:<30} → {e['label']:<25} ({e['confidence']:.2f})")
        return "\n".join(lines)


PDF_PATH = "pii_test_document.pdf"


def load_pdf(path: str) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Installer pypdf: pip install pypdf")

    full_text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    # Split på sætningsafslutning og filtrer for korte fragmenter
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    return [s.strip().replace("\n", " ") for s in sentences if len(s.strip()) > 10]


if __name__ == "__main__":
    detector = PIIDetector()
    sentences = load_pdf(PDF_PATH)

    print(f"\nIndlæste {len(sentences)} sætninger fra '{PDF_PATH}'")
    print("═" * 60)

    for sentence in sentences:
        result = detector.predict(sentence)
        if result.has_pii:
            print(result)
            print("─" * 60)

    print(f"\nFærdig – {sum(1 for s in sentences if detector.predict(s).has_pii)} / {len(sentences)} sætninger indeholdt PII.")