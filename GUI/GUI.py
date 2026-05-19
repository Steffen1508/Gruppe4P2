import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from pypdf import PdfReader
from pii_pipeline import run_pii_detection


# ============================================================
# PDF-logikk
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Leser tekst fra en PDF-fil og returnerer teksten som én samlet streng.

    Denne funksjonen er bevisst holdt utenfor GUI-klassene slik at PDF-logikk
    ikke blandes direkte inn i brukergrensesnittet.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError("PDF-filen finnes ikke.")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Ugyldig filtype. Kun PDF-filer er støttet.")

    try:
        reader = PdfReader(str(path))
        pages_text = []

        for page in reader.pages:
            # extract_text() kan returnere None hvis siden ikke inneholder lesbar tekst.
            text = page.extract_text() or ""
            pages_text.append(text)

        full_text = "\n\n".join(pages_text).strip()

        if not full_text:
            raise ValueError(
                "Fant ingen lesbar tekst i PDF-en. "
                "PDF-en kan være skannet som bilde og kreve OCR."
            )

        return full_text

    except Exception as error:
        raise RuntimeError(f"Feil ved PDF-lesing: {error}") from error


# ============================================================
# Venstre panel: input
# ============================================================

class InputPanel(QFrame):
    """
    Panel for manuell tekstinput og dra-og-slipp av PDF.

    Panelet arver fra QFrame for å kunne styles tydelig som en egen seksjon.
    """

    def __init__(self):
        super().__init__()

        self.is_loading_pdf = False
        self.input_source = "manual"

        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.title_label = QLabel("Input")
        self.title_label.setObjectName("PanelTitle")

        self.source_label = QLabel("Source: Manual text")
        self.source_label.setObjectName("SourceLabel")

        self.help_label = QLabel("Write/paste text below, or drag and drop a PDF into this panel.")
        self.help_label.setWordWrap(True)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Write or paste text here...")
        self.text_edit.textChanged.connect(self.mark_as_manual_input)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.source_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def mark_as_manual_input(self):
        """
        Oppdaterer input-kilden når brukeren skriver manuelt.

        Når vi fyller tekstfeltet programmatisk etter PDF-drop, bruker vi
        self.is_loading_pdf for å unngå at kilden feilaktig blir satt til manuell.
        """
        if not self.is_loading_pdf:
            self.input_source = "manual"
            self.source_label.setText("Source: Manual text")

    def get_input_text(self) -> str:
        """Returnerer teksten som skal sendes til PII-deteksjon."""
        return self.text_edit.toPlainText().strip()

    def dragEnterEvent(self, event):
        """
        Godtar drag-enter dersom brukeren drar inn en lokal PDF-fil.
        """
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()

            if urls and urls[0].isLocalFile():
                file_path = Path(urls[0].toLocalFile())

                if file_path.suffix.lower() == ".pdf":
                    event.acceptProposedAction()
                    return

        event.ignore()

    def dropEvent(self, event):
        """
        Håndterer faktisk filslipp.

        Leser PDF-tekst og setter den inn i tekstfeltet.
        """
        urls = event.mimeData().urls()

        if not urls:
            QMessageBox.warning(self, "Invalid drop", "No file was dropped.")
            return

        file_path = Path(urls[0].toLocalFile())

        if file_path.suffix.lower() != ".pdf":
            QMessageBox.warning(
                self,
                "Invalid file type",
                "Only PDF files are supported.",
            )
            return

        try:
            extracted_text = extract_text_from_pdf(str(file_path))

            self.is_loading_pdf = True
            self.text_edit.setPlainText(extracted_text)
            self.is_loading_pdf = False

            self.input_source = "pdf"
            self.source_label.setText(f"Source: PDF – {file_path.name}")

            event.acceptProposedAction()

        except Exception as error:
            self.is_loading_pdf = False
            QMessageBox.critical(
                self,
                "PDF reading error",
                str(error),
            )


# ============================================================
# Midtre panel: prosessering
# ============================================================

class ProcessingPanel(QFrame):
    """
    Panel med Detect-knapp og latency-visning.
    """

    def __init__(self):
        super().__init__()

        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.title_label = QLabel("Processing")
        self.title_label.setObjectName("PanelTitle")

        self.detect_button = QPushButton("Detect")
        self.detect_button.setMinimumHeight(60)
        self.detect_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.latency_label = QLabel("Latency: -")
        self.latency_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.detect_button)
        layout.addWidget(self.latency_label)
        layout.addStretch()

        self.setLayout(layout)

    def set_latency(self, total_ms: float, avg_chunk_ms: float | None = None):
        def fmt(ms: float) -> str:
            return f"{ms:.1f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"

        if avg_chunk_ms is not None:
            self.latency_label.setText(f"Total: {fmt(total_ms)} | Avg/chunk: {fmt(avg_chunk_ms)}")
        else:
            self.latency_label.setText(f"Latency: {fmt(total_ms)}")

    def reset_latency(self):
        self.latency_label.setText("Latency: -")


# ============================================================
# Høyre panel: output
# ============================================================

class OutputPanel(QFrame):
    """
    Panel som viser resultatet fra PII-deteksjonen.
    """

    def __init__(self):
        super().__init__()

        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.title_label = QLabel("Output")
        self.title_label.setObjectName("PanelTitle")

        self.status_label = QLabel("No detection run yet")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Detected text", "PII category", "Confidence"])
        self.result_table.horizontalHeader().setFixedHeight(50)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_table)

        self.setLayout(layout)

    def show_results(self, detections: list[dict[str, str]]):
        """
        Viser PII-resultater i tabellen.
        """
        self.result_table.setRowCount(0)

        if not detections:
            self.status_label.setText("No PII detected")
            self.status_label.setProperty("state", "clean")
            self.refresh_style()
            return

        self.status_label.setText("PII detected")
        self.status_label.setProperty("state", "warning")
        self.refresh_style()

        self.result_table.setRowCount(len(detections))

        for row, item in enumerate(detections):
            detected_text = item.get("text", "")
            category = item.get("category", "unknown")
            confidence = item.get("confidence", None)
            confidence_str = f"{confidence:.2%}" if confidence is not None else "-"

            self.result_table.setItem(row, 0, QTableWidgetItem(detected_text))
            self.result_table.setItem(row, 1, QTableWidgetItem(category))
            self.result_table.setItem(row, 2, QTableWidgetItem(confidence_str))

    def show_error(self, message: str):
        """
        Viser prosesseringsfeil i output-panelet.
        """
        self.result_table.setRowCount(0)
        self.status_label.setText(f"Error: {message}")
        self.status_label.setProperty("state", "error")
        self.refresh_style()

    def refresh_style(self):
        """
        Tvinger Qt til å oppdatere stylesheet etter at vi har endret property.
        """
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)


# ============================================================
# Hovedvindu
# ============================================================

class MainWindow(QMainWindow):
    """
    Hovedvinduet som setter sammen de tre panelene:

    1. Venstre: input
    2. Midten: prosessering
    3. Høyre: output
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PII Detector")
        self.setGeometry(300, 200, 1200, 600)

        self.input_panel = InputPanel()
        self.processing_panel = ProcessingPanel()
        self.output_panel = OutputPanel()

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """
        Bygger hovedlayouten med tre horisontale paneler.
        """
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        main_layout.addWidget(self.input_panel, stretch=3)
        main_layout.addWidget(self.processing_panel, stretch=1)
        main_layout.addWidget(self.output_panel, stretch=3)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.apply_styles()

    def connect_signals(self):
        """
        Kobler knappen i prosesseringspanelet til deteksjonsmetoden.
        """
        self.processing_panel.detect_button.clicked.connect(self.detect_pii)

    def detect_pii(self):
        """
        Leser input, kjører PII-deteksjon, måler latency og viser resultat.
        """
        input_text = self.input_panel.get_input_text()

        if not input_text:
            QMessageBox.warning(
                self,
                "Missing input",
                "Please enter text or drop a PDF before running detection.",
            )
            return

        self.processing_panel.detect_button.setEnabled(False)
        self.processing_panel.detect_button.setText("Detecting...")
        self.processing_panel.reset_latency()

        # Gjør at GUI-et rekker å oppdatere knappetekst før prosessering starter.
        QApplication.processEvents()

        try:
            detections, total_ms, avg_chunk_ms = run_pii_detection(input_text)

            self.processing_panel.set_latency(total_ms, avg_chunk_ms)
            self.output_panel.show_results(detections)

        except Exception as error:
            self.processing_panel.reset_latency()
            self.output_panel.show_error(str(error))

            QMessageBox.critical(
                self,
                "PII processing error",
                f"An error occurred during PII processing:\n\n{error}",
            )

        finally:
            self.processing_panel.detect_button.setEnabled(True)
            self.processing_panel.detect_button.setText("Detect")

    def apply_styles(self):
        """
        Enkel styling for å gjøre GUI-et ryddigere og tydeligere.
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f4f4;
            }

            QFrame {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 8px;
            }

            QLabel#PanelTitle {
                font-size: 20px;
                font-weight: bold;
                padding-bottom: 8px;
            }

            QLabel#SourceLabel {
                color: #555555;
                font-style: italic;
            }

            QLabel#StatusLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }

            QLabel#StatusLabel[state="warning"] {
                color: #8a4b00;
                background-color: #fff3cd;
                border: 1px solid #ffecb5;
            }

            QLabel#StatusLabel[state="clean"] {
                color: #0f5132;
                background-color: #d1e7dd;
                border: 1px solid #badbcc;
            }

            QLabel#StatusLabel[state="error"] {
                color: #842029;
                background-color: #f8d7da;
                border: 1px solid #f5c2c7;
            }

            QTextEdit {
                font-size: 14px;
                border: 1px solid #bbbbbb;
                border-radius: 6px;
                padding: 8px;
            }

            QPushButton {
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px;
                background-color: #2d6cdf;
                color: white;
            }

            QPushButton:hover {
                background-color: #1f5cc5;
            }

            QPushButton:disabled {
                background-color: #999999;
            }

            QTableWidget {
                border: 1px solid #bbbbbb;
                border-radius: 6px;
                gridline-color: #dddddd;
                font-size: 13px;
            }

            QHeaderView::section {
                font-weight: bold;
                padding: 6px;
                background-color: #eeeeee;
                border: 1px solid #cccccc;
            }
        """)


# ============================================================
# Programstart
# ============================================================

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()