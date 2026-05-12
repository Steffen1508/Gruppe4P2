from __future__ import annotations

import time

from BERT_inference import MAX_LEN, PIIDetector, PIIResult
from text_preparation import chunk_with_offsets, count_tokens


class PiiPipeline:
    # Reservér plads til [CLS] og [SEP] — effektiv max vi kan sende til predict()
    EFFECTIVE_MAX = MAX_LEN - 2

    def __init__(self, overlap: int = 32) -> None:
        self.overlap = overlap
        self._detector = None

    def _ensure_loaded(self) -> None:
        if self._detector == None:
            self._detector = PIIDetector()

    def predict(self, text: str) -> PIIResult:
        if not text or not text.strip():
            raise ValueError("Tom tekst – indtast venligst noget tekst.")

        self._ensure_loaded()
        n_tokens = count_tokens(text, self._detector.tokenizer)

        # Kort nok til at køre direkte uden chunking
        if n_tokens <= self.EFFECTIVE_MAX:
            return self._detector.predict(text)

        return self._predict_chunked(text)

    def _predict_chunked(self, text: str) -> PIIResult:
        start_time = time.perf_counter()
        chunks = chunk_with_offsets(text, self._detector.tokenizer, max_tokens=MAX_LEN, overlap=self.overlap)

        all_entities = []
        seen = set()  # dedup-nøgle: (start, end, label) i original tekst

        for chunk_text, char_offset in chunks:
            try:
                chunk_result = self._detector.predict(chunk_text)
            except ValueError:
                continue  # tomme chunks springes over

            for ent in chunk_result.entities:
                # Oversæt chunk-koordinater til original-koordinater
                abs_start = ent["start"] + char_offset
                abs_end = ent["end"] + char_offset
                key = (abs_start, abs_end, ent["label"])
                if key in seen:
                    continue
                seen.add(key)
                all_entities.append({
                    "text":       text[abs_start:abs_end],
                    "label":      ent["label"],
                    "confidence": ent["confidence"],
                    "start":      abs_start,
                    "end":        abs_end,
                })

        all_entities.sort(key=lambda e: (e["start"], e["end"]))
        latency_ms = (time.perf_counter() - start_time) * 1000
        return PIIResult(text=text, entities=all_entities, latency_ms=latency_ms)


_pipeline = None


def run_pii_detection(input_text: str) -> list[dict]:
    global _pipeline
    if _pipeline == None:
        _pipeline = PiiPipeline()
    result = _pipeline.predict(input_text)
    # Mapper 'label' til 'category' som GUI'en forventer
    return [
        {
            "text":       ent["text"],
            "category":   ent["label"],
            "confidence": ent["confidence"],
            "start":      ent["start"],
            "end":        ent["end"],
        }
        for ent in result.entities
    ]


def get_pipeline() -> PiiPipeline:
    # Returnerer det fulde PIIResult med latency_ms og confidence_avg
    global _pipeline
    if _pipeline == None:
        _pipeline = PiiPipeline()
    return _pipeline


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Ring til Jonas Hansen på jonas.hansen@example.com eller +45 12 34 56 78. "
        "Hans IBAN er DK50 0040 0440 1162 43 og kortet ender på 4242 4242 4242 4242."
    )

    print(f"Input:\n{sample}\n")
    print("Loader model og kører pipeline...")
    detections = run_pii_detection(sample)
    print(f"\nFandt {len(detections)} PII-entiteter:")
    for d in detections:
        print(f"  - {d['category']:25s} → {d['text']!r}  (conf {d['confidence']:.2f})")