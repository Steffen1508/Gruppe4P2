# Model kan downloades her: https://www.dropbox.com/t/yhU1bqLll8bMo166
#
# BERT_inference.py
# ─────────────────────────────────────────────
# Indlæser den trænede PII-model og kører inference.
# Bruges som udgangspunkt for det endelige system.
# ─────────────────────────────────────────────

import torch
import time
from transformers import BertTokenizer, BertForTokenClassification

# ═══════════════════════════════════════════════════════════════════
# INDSTILLINGER
# ═══════════════════════════════════════════════════════════════════

# Sti til den gemte model (samme som save_path i træningsscriptet)
MODEL_PATH = "saved_model_reduced"

# Maks antal tokens per tekst – skal matche det der blev brugt under træning
MAX_LEN = 128

# Enhed – "cuda" hvis GPU er tilgængelig, ellers "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════


# Labels – skal matche label_map fra BERT_imp_v4.py
label_map = {
    "O": 0,

    # Kategori 1 - Højest prioritet (finansielt/adgang)
    "API_KEY": 1,
    "CREDIT_CARD_NUMBER": 2,
    "BANK_ACCOUNT_NUMBER": 3,
    "IBAN": 4,

    # Kategori 2 - Høj prioritet (identitet/adgang)
    "PASSWORD": 5,
    "PASSPORT_NUMBER": 6,
    "SSN": 7,

    # Kategori 3 - Medium prioritet (personlig info)
    "FULL_NAME": 8,
    "FIRST_NAME": 9,
    "LAST_NAME": 10,
    "EMAIL": 11,
    "PHONE_NUMBER": 12,
}

INV_label_map = {v: k for k, v in label_map.items()}


class PIIDetector:
    """
    Indlæser en gemt BERT-model og kører PII-detektion på tekst.

    Eksempel:
        detector = PIIDetector()
        result = detector.predict("Ring til Jonas Hansen på Jonas@gmail.com")
        print(result.entities)   # [{"text": "Jonas Hansen", "label": "FULL_NAME", ...}, ...]
        print(result.has_pii)    # True
    """

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"Indlæser model fra: {model_path}")
        self.device    = DEVICE
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model     = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model klar – kører på: {self.device}")

        # Warm-up kald så første rigtige forespørgsel ikke rammes af cold start
        self.predict("warmup")

    def predict(self, text: str) -> "PIIResult":
        """
        Analyserer en tekst og returnerer et PIIResult objekt.

        Args:
            text: Den tekst der skal analyseres

        Returns:
            PIIResult med entities, has_pii, confidence scores og latency
        """
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

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Softmax for at få confidence scores
        probs       = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
        predictions = probs.argmax(dim=-1)

        # Saml tokens og labels
        tokens      = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        mask        = attention_mask[0].cpu()

        raw_tokens = []
        for idx, (token, pred, active, offset) in enumerate(zip(tokens, predictions, mask, offset_mapping)):
            if not active or token in ("[CLS]", "[SEP]", "[PAD]"):
                continue

            label      = INV_label_map[pred.item()]
            confidence = probs[idx][pred.item()].item()

            raw_tokens.append({
                "token":      token,
                "label":      label,
                "confidence": confidence,
                "start":      offset[0].item(),
                "end":        offset[1].item(),
            })

        # Saml WordPiece-tokens og sammenhængende entiteter
        entities = self._merge_entities(raw_tokens, text)

        # Post-processing: fix emails og passwords der er splittet af modellen
        entities = self._fix_split_entities(entities, text)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return PIIResult(text=text, entities=entities, latency_ms=latency_ms)

    def _fix_split_entities(self, entities: list, text: str) -> list:
        """
        Post-processing der fixer to kendte problemer:

        1. Emails splittet af modellen (f.eks. "kevin.larsen@" + ".net"):
           Finder den fulde email i den originale tekst via regex og
           erstatter de splittede fragmenter med én samlet entitet.

        2. Passwords der starter midt i et ord (f.eks. "!Secure42" i stedet
           for "Password!Secure42"): Udvider baglæns til første mellemrum.
        """
        EMAIL_RE    = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
        PASSWORD_RE = re.compile(r'\S+')  # Alt uden mellemrum

        fixed = []
        used  = set()

        for i, ent in enumerate(entities):
            if i in used:
                continue

            # ── Fix splittede emails ──────────────────────────────
            if ent["label"] == "EMAIL":
                # Find den fulde email i den originale tekst tæt på denne position
                search_start = max(0, ent["start"] - 20)
                search_end   = min(len(text), ent["end"] + 60)
                snippet      = text[search_start:search_end]

                match = EMAIL_RE.search(snippet)
                if match:
                    abs_start = search_start + match.start()
                    abs_end   = search_start + match.end()
                    full_email = text[abs_start:abs_end]

                    # Marker alle efterfølgende EMAIL-fragmenter der overlapper
                    for j, other in enumerate(entities):
                        if j != i and other["label"] == "EMAIL":
                            if other["start"] >= abs_start and other["end"] <= abs_end + 5:
                                used.add(j)

                    fixed.append({
                        "text":       full_email,
                        "label":      "EMAIL",
                        "confidence": ent["confidence"],
                        "start":      abs_start,
                        "end":        abs_end,
                    })
                    continue

            # ── Fix passwords der starter midt i et ord ──────────
            if ent["label"] == "PASSWORD":
                # Find starten af det fulde ord (bagud til mellemrum)
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
        """
        Slår sammenhængende tokens med samme label sammen til én entitet.
        Bruger offset-positioner fra original tekst til rekonstruktion,
        så emails og andre tokens med punktum/bindestreg ikke splittes forkert.
        """
        if not raw_tokens:
            return []

        entities = []
        current  = None

        for token in raw_tokens:
            label = token["label"]

            # WordPiece ## tokens – altid del af forrige token
            if token["token"].startswith("##"):
                if current and current["label"] == label:
                    current["end"]        = token["end"]
                    current["confidence"] = min(current["confidence"], token["confidence"])
                    # Rekonstruer tekst fra original streng via offsets
                    current["text"] = text[current["start"]:current["end"]]
                continue

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            # Forskellig label fra forrige – gem og start forfra
            if current and current["label"] != label:
                entities.append(current)
                current = None

            if current and current["label"] == label:
                # Udvid kun hvis tokens er tæt på hinanden (maks 2 tegn mellemrum)
                # Det fanger f.eks. "kevin.larsen @ corp.net" men ikke "name ... email"
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


class PIIResult:
    """
    Resultat fra PIIDetector.predict().

    Attributter:
        text           : Den originale inputtekst
        entities       : Liste af fundne PII-entiteter
        has_pii        : True hvis der blev fundet PII
        latency_ms     : Målt svartid i millisekunder (NFR1, NFR3)
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
        status = "✓ PII FUNDET" if self.has_pii else "✗ Ingen PII"
        lines  = [
            f"[{status}] '{self.text}'",
            f"  Svartid: {self.latency_ms:.1f} ms  |  "
            f"Gns. confidence: {self.confidence_avg:.2f}",
        ]
        if self.has_pii:
            lines.append("  " + "─" * 55)
            for e in self.entities:
                lines.append(
                    f"  {e['text']:<30} → {e['label']:<25} ({e['confidence']:.2f})"
                )
        return "\n".join(lines)


import re

# ─────────────────────────────────────────────
# Demo – kør scriptet direkte for at teste
# ─────────────────────────────────────────────

PDF_PATH = "pii_test_document.pdf"

def load_pdf(path: str) -> list[str]:
    """
    Indlæser en PDF og returnerer en liste af sætninger.
    Kræver: pip install pypdf
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Installer pypdf: pip install pypdf")

    reader    = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # Split på punktum, udråbstegn og spørgsmålstegn – behold ikke-tomme sætninger
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip().replace("\n", " ") for s in sentences if len(s.strip()) > 10]
    return sentences


if __name__ == "__main__":
    detector  = PIIDetector()
    sentences = load_pdf(PDF_PATH)

    print(f"\nIndlæste {len(sentences)} sætninger fra '{PDF_PATH}'")
    print("═" * 60)

    for sentence in sentences:
        result = detector.predict(sentence)
        if result.has_pii:
            print(result)
            print("─" * 60)

    print(f"\nFærdig – {sum(1 for s in sentences if detector.predict(s).has_pii)} / {len(sentences)} sætninger indeholdt PII.")