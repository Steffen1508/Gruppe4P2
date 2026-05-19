"""
iou_compound_test.py
────────────────────────────────────────────────────────────────────
Tester om BERT-modellen finder sammensatte (multi-token) PII-entiteter
og evaluerer med IOU (Intersection over Union) i stedet for exact match.

Baggrund
────────
Exact match giver 0 hvis modellen finder "Lars" i stedet for
"Lars Peter Nielsen". IOU = overlap / union giver ~0.24 — mere
informativt. Dette script viser præcis hvor modellen fejler på
multi-ord entiteter og om IOU-baseret evaluering er mere retfærdig.

Kør:               python iou_compound_test.py
Generer PDF:       python iou_compound_test.py --generate-pdf
Anden model-sti:   python iou_compound_test.py --model ../GUI/saved_model_combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Model-sti: prøv GUI/saved_model_combined hvis lokal ikke findes ──
_DEFAULT_MODEL = Path(__file__).parent / "saved_model_combined"
_FALLBACK_MODEL = Path(__file__).parent.parent / "GUI" / "saved_model_combined"

# ── Testdokument (holdes under ~350 BERT-tokens for at undgå truncation) ───

TEST_TEXT = """\
FORTROLIGT TEST – IOU-evaluering af sammensatte PII-entiteter

=== Navne (multi-token – IOU-udfordring) ===
Lars Peter Nielsen er registreret som primær kontoejer.
Anne-Marie Christoffersen er juridisk repræsentant i sagen.
Mohammed Al-Hassan har indsendt en formel klage.
Mette Frederiksen-Larsen er sekundær kontaktperson.
Hans Christian Bjørnsson underskrev kontrakten den 3. januar 2024.
Fatima El-Zahrawi er tilknyttet som ekstern konsulent.
Mikkel Johannes Andersen er klager i sag 2024-447.
Anna Sofie Kjærgaard bekræftede sine oplysninger telefonisk.

=== Kontaktoplysninger ===
E-mail: lars.peter.nielsen@gmail.com  Tlf: +45 31 45 67 89
E-mail: am.christoffersen@nordea.dk   Tlf: 61 23 45 67
E-mail: m.al-hassan@company.dk        Tlf: +45 70 22 33 44

=== Finansielle oplysninger ===
IBAN (Lars Peter Nielsen):         DK89 3000 0009 1234 56
IBAN (Mikkel Johannes Andersen):   DK50 0040 0440 1162 43
Kort (Anna Sofie Kjærgaard):       4111 1111 1111 1111
Kort (Lars Peter Nielsen):         5425 2334 3010 9903

=== Adgangskoder og nøgler ===
Password (Lars Peter Nielsen): Sommer2024!
Password (Mikkel Johannes Andersen): MyP@ssw0rd#2024
API-nøgle: sk-prod-Hj8kL2mN9pQr3sT5uV7wX1yZ
CPR (Lars Peter Nielsen): 140382-1234
Pas (Fatima El-Zahrawi): EU123456789
"""

# ── Ground truth ─────────────────────────────────────────────────────────────
# (fragment, label) – build_ground_truth finder ALLE forekomster i TEST_TEXT.
# Navne gentages i finansielle/kode-sektioner → model testes i flere kontekster.

GROUND_TRUTH_SPECS: list[tuple[str, str]] = [
    # Multi-token navne (IOU-kerneproblem)
    ("Lars Peter Nielsen",        "FULL_NAME"),
    ("Anne-Marie Christoffersen", "FULL_NAME"),
    ("Mohammed Al-Hassan",        "FULL_NAME"),
    ("Mette Frederiksen-Larsen",  "FULL_NAME"),
    ("Hans Christian Bjørnsson",  "FULL_NAME"),
    ("Fatima El-Zahrawi",         "FULL_NAME"),
    ("Mikkel Johannes Andersen",  "FULL_NAME"),
    ("Anna Sofie Kjærgaard",      "FULL_NAME"),
    # E-mails
    ("lars.peter.nielsen@gmail.com", "EMAIL"),
    ("am.christoffersen@nordea.dk",  "EMAIL"),
    ("m.al-hassan@company.dk",       "EMAIL"),
    # Telefoner (med mellemrum → IOU-problem ved splittet tokenisering)
    ("+45 31 45 67 89", "PHONE_NUMBER"),
    ("61 23 45 67",     "PHONE_NUMBER"),
    ("+45 70 22 33 44", "PHONE_NUMBER"),
    # IBAN (med mellemrum)
    ("DK89 3000 0009 1234 56", "IBAN"),
    ("DK50 0040 0440 1162 43", "IBAN"),
    # Kreditkort (med mellemrum)
    ("4111 1111 1111 1111", "CREDIT_CARD_NUMBER"),
    ("5425 2334 3010 9903", "CREDIT_CARD_NUMBER"),
    # Passwords
    ("Sommer2024!",     "PASSWORD"),
    ("MyP@ssw0rd#2024", "PASSWORD"),
    # API-nøgle
    ("sk-prod-Hj8kL2mN9pQr3sT5uV7wX1yZ", "API_KEY"),
    # Identifikation
    ("140382-1234", "SSN"),
    ("EU123456789", "PASSPORT_NUMBER"),
]


# ── IOU-logik ────────────────────────────────────────────────────────────────

def build_ground_truth(text: str, specs: list[tuple[str, str]]) -> list[dict]:
    gt = []
    for fragment, label in specs:
        pos = 0
        while True:
            idx = text.find(fragment, pos)
            if idx == -1:
                break
            gt.append({"text": fragment, "label": label, "start": idx, "end": idx + len(fragment)})
            pos = idx + len(fragment)
    return sorted(gt, key=lambda x: x["start"])


def span_iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def match_gt_to_predictions(ground_truth: list[dict], predictions: list[dict]) -> list[dict]:
    results = []
    for gt in ground_truth:
        best_iou, best_pred = 0.0, None
        for pred in predictions:
            iou = span_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_pred = iou, pred
        results.append({
            "gt":            gt,
            "pred":          best_pred,
            "iou":           best_iou,
            "exact":         best_iou == 1.0,
            "partial":       0.0 < best_iou < 1.0,
            "missed":        best_iou == 0.0,
            "label_correct": best_pred is not None and best_pred.get("label") == gt["label"],
        })
    return results


# ── Rapport ──────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    exact   = [r for r in results if r["exact"]]
    partial = [r for r in results if r["partial"]]
    missed  = [r for r in results if r["missed"]]
    total   = len(results)
    avg_iou = sum(r["iou"] for r in results) / total if total else 0.0

    W = 70
    print("\n" + "═" * W)
    print("IOU EVALUERINGSRAPPORT – Sammensatte PII-entiteter")
    print("═" * W)
    print(f"Totale GT-entiteter : {total}")
    print(f"Exact match (IOU=1) : {len(exact):3d}  ({100*len(exact)/total:.1f}%)")
    print(f"Partial (0<IOU<1)   : {len(partial):3d}  ({100*len(partial)/total:.1f}%)")
    print(f"Missed (IOU=0)      : {len(missed):3d}  ({100*len(missed)/total:.1f}%)")
    print(f"Gennemsnit IOU      : {avg_iou:.3f}")

    # Per-label breakdown
    by_label: dict[str, list] = {}
    for r in results:
        by_label.setdefault(r["gt"]["label"], []).append(r)

    print("\n── Per label " + "─" * (W - 13))
    header = f"{'Label':<25} {'GT':>4} {'Exact':>6} {'Partial':>8} {'Missed':>7} {'Avg IOU':>8}"
    print(header)
    print("─" * W)
    for lbl, items in sorted(by_label.items()):
        n   = len(items)
        e   = sum(1 for x in items if x["exact"])
        p   = sum(1 for x in items if x["partial"])
        m   = sum(1 for x in items if x["missed"])
        avg = sum(x["iou"] for x in items) / n
        print(f"{lbl:<25} {n:>4} {e:>6} {p:>8} {m:>7} {avg:>8.3f}")

    # Detaljerede partial matches
    if partial:
        print(f"\n── Partial matches (model fandt kun en del) " + "─" * (W - 43))
        for r in partial:
            gt, pred = r["gt"], r["pred"]
            lbl_ok = "✓" if r["label_correct"] else "✗ label fejl"
            print(f"  GT  : '{gt['text']}'  ({gt['label']})")
            print(f"  Pred: '{pred['text']}'  ({pred.get('label','?')})  IOU={r['iou']:.3f}  {lbl_ok}")
            print()

    if missed:
        print(f"── Missed (model fandt intet) " + "─" * (W - 30))
        for r in missed:
            g = r["gt"]
            print(f"  MISSED: '{g['text']}'  ({g['label']})")

    print("\n── Konklusion " + "─" * (W - 14))
    if avg_iou >= 0.9:
        print("  Modellen finder sammensatte entiteter meget godt (IOU ≥ 0.9).")
        print("  Exact match og IOU giver sandsynligvis sammenlignelige resultater.")
    elif avg_iou >= 0.6:
        print("  Modellen har delvis succes med sammensatte entiteter (IOU 0.6–0.9).")
        print("  IOU-baseret evaluering ville give en mere retfærdig score end exact match.")
    else:
        print("  Modellen kæmper med sammensatte entiteter (IOU < 0.6).")
        print("  Exact match undervurderer modellens præstation markant.")
        print("  Anbefaling: Brug IOU > 0.5 som match-kriterium i evalueringen.")
    print("═" * W + "\n")


# ── PDF-generator ─────────────────────────────────────────────────────────────

def generate_pdf(output_path: str) -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        txt_path = output_path.replace(".pdf", ".txt")
        print(f"[!] fpdf2 ikke installeret (pip install fpdf2).")
        print(f"    Gemmer testdokumentet som tekst: {txt_path}")
        Path(txt_path).write_text(TEST_TEXT, encoding="utf-8")
        return

    # Helvetica understøtter kun latin-1 – erstat ikke-understøttede tegn
    _REPLACEMENTS = str.maketrans({
        "–": "-", "—": "-", "═": "=", "─": "-",
        "≥": ">=", "≤": "<=", "→": "->", "✓": "v",
        "►": ">", "·": ".", "•": "*",
    })

    def _clean(text: str) -> str:
        return text.translate(_REPLACEMENTS)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pw = pdf.epw  # effective page width (minus margins)
    for line in TEST_TEXT.splitlines():
        pdf.set_x(pdf.l_margin)
        if line.startswith("==="):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(pw, 7, _clean(line.strip("= ")), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
        elif not line.strip():
            pdf.ln(3)
        else:
            pdf.multi_cell(pw, 5, _clean(line))

    pdf.output(output_path)
    print(f"PDF genereret: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate-pdf", action="store_true",
                        help="Generer test-PDF og afslut (kræver fpdf2)")
    parser.add_argument("--pdf-path", default=str(Path(__file__).parent / "iou_test_document.pdf"),
                        help="Outputsti til PDF (standard: BERT/iou_test_document.pdf)")
    parser.add_argument("--model", default=None,
                        help="Sti til saved_model-mappen (standard: forsøger GUI/saved_model_combined)")
    args = parser.parse_args()

    if args.generate_pdf:
        generate_pdf(args.pdf_path)
        return

    # Bestem model-sti
    if args.model:
        model_path = args.model
    elif _DEFAULT_MODEL.exists():
        model_path = str(_DEFAULT_MODEL)
    elif _FALLBACK_MODEL.exists():
        model_path = str(_FALLBACK_MODEL)
        print(f"[i] Bruger model fra GUI-mappen: {model_path}")
    else:
        print("[!] Ingen model fundet. Angiv sti med --model <sti>")
        sys.exit(1)

    # Tilføj BERT-mappen til sys.path så BERT_inference kan importeres
    sys.path.insert(0, str(Path(__file__).parent))
    from BERT_inference import PIIDetector

    print("Bygger ground truth annotationer...")
    ground_truth = build_ground_truth(TEST_TEXT, GROUND_TRUTH_SPECS)
    print(f"  {len(ground_truth)} GT-entiteter fundet i testdokumentet")

    # Vis token-antal som en hurtigt sanity-check
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(model_path)
    n_tokens = len(tok.encode(TEST_TEXT))
    print(f"  Testdokument: {len(TEST_TEXT)} tegn  /  {n_tokens} BERT-tokens", end="")
    if n_tokens > 510:
        print("  ⚠ OVER 512 – teksten vil blive trunceret!", end="")
    print()

    print("\nLoader BERT-model...")
    detector = PIIDetector(model_path=model_path)

    print("\nKører inference...")
    result = detector.predict(TEST_TEXT)
    predictions = result.entities
    print(f"  {len(predictions)} entiteter fundet af modellen")

    matched = match_gt_to_predictions(ground_truth, predictions)
    print_report(matched)


if __name__ == "__main__":
    main()
