# Hvis kode skal køres skal der først installeres disse pakker: pip install transformers torch tqdm

import pandas as pd
from typing import Tuple, List
import re
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. PIIDataset
# ─────────────────────────────────────────────

class PIIDataset(Dataset):
    """
    Forbereder vores tekst og labels så BERT kan læse dem.

    BERT kan ikke arbejde med rå tekst og strenge som "PII".
    Den forventer tal – derfor konverterer denne klasse alt til tensors.

    Args:
        texts     : Liste af rå inputtekster fra source_text kolonnen
        labels    : Liste af labels - enten "PII" eller "NON-PII"
        tokenizer : BERTs tokenizer, som oversætter ord til tal
        max_len   : Hvor mange tokens må en tekst maks fylde (default 128)
    """

    # BERT arbejder med tal, ikke tekst - så vi laver en simpel oversættelse
    LABEL_MAP = {"NON-PII": 0, "PII": 1}

    def __init__(self, texts: list, labels: list, tokenizer: BertTokenizer, max_len: int = 128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        # PyTorch kalder denne for at vide hvor stort datasættet er
        return len(self.texts)

    def __getitem__(self, idx):
        # Hent tekst og label for ét eksempel
        text  = self.texts[idx]
        label = self.labels[idx]

        # Tokenizeren opdeler teksten i tokens og returnerer tre ting:
        #   input_ids      – hvert ord/token som et tal i BERTs vokabular
        #   attention_mask – 1 for rigtige tokens, 0 for tom padding
        #
        # padding="max_length" fylder korte tekster op til max_len
        # truncation=True      klipper tekster der er længere end max_len
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # .squeeze(0) fjerner en unødvendig ekstra dimension som tokenizeren tilføjer
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Oversæt "PII"/"NON-PII" til 1/0 som en tensor
        label_tensor = torch.tensor(self.LABEL_MAP[label], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         label_tensor,
        }


# ─────────────────────────────────────────────
# 2. BertTrainer
# ─────────────────────────────────────────────

class BertTrainer:
    """
    Håndterer alt det BERT-relaterede: model, træning og evaluering.

    Arbejdsgangen er:
        1. Indlæs den præ-trænede BERT model fra Hugging Face
        2. Træn den på vores PII data (fine-tuning)
        3. Evaluer hvor godt den klarer sig på testdata
    """

    def __init__(self, model_name: str = "bert-base-uncased", epochs: int = 3,
                 batch_size: int = 16, lr: float = 2e-5, max_len: int = 128):

        # Gem indstillinger så vi kan bruge dem i de andre metoder
        self.model_name = model_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.max_len    = max_len

        # Brug GPU hvis den findes, ellers CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Bruger device: {self.device}")

        # Indlæs BERTs tokenizer – den skal matche model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Indlæs selve BERT modellen med et klassifikationslag ovenpå (2 klasser: PII / NON-PII)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

    def _make_dataloader(self, texts: list, labels: list, shuffle: bool) -> DataLoader:
        """
        Hjælpemetode der pakker tekster og labels ind i en DataLoader.
        DataLoader sørger for at fodre modellen med ét batch ad gangen.
        """
        dataset = PIIDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, X_train: list, y_train: list, X_val: list, y_val: list):
        """
        Fine-tuner BERT på vores træningsdata.

        For hvert epoch:
            1. Kør alle træningsbatches og opdater modellens vægte
            2. Evaluer på valideringsdata for at følge med i fremgangen
        """

        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader   = self._make_dataloader(X_val,   y_val,   shuffle=False)

        # AdamW er den anbefalede optimizer til BERT fine-tuning
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # Scheduleren sænker gradvist learning rate undervejs i træningen
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self._evaluate(val_loader)

            print(f"Epoch {epoch}/{self.epochs}  |  "
                  f"Train loss: {train_loss:.4f}  |  "
                  f"Val loss: {val_loss:.4f}  |  "
                  f"Val accuracy: {val_acc:.4f}")

    def _train_one_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        """
        Kører ét fuldt gennemløb af træningsdataen.
        Returnerer gennemsnitligt tab (loss) for epochen.
        """
        self.model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc="  Træner", unit="batch", leave=False)
        for batch in progress:
            # Nulstil gradienter fra forrige batch
            optimizer.zero_grad()

            # Flyt batch til GPU/CPU
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            # Forlæns pas: beregn loss
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Baglæns pas: beregn gradienter og opdater vægte
            output.loss.backward()

            # Gradient clipping forhindrer at vægtene hopper for vildt
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += output.loss.item()

            # Vis løbende loss i progress baren
            progress.set_postfix(loss=f"{output.loss.item():.4f}")

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluerer modellen på et datasæt uden at opdatere vægte.
        Returnerer gennemsnitligt tab og accuracy.
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        # torch.no_grad() sparer hukommelse når vi ikke træner
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluerer", unit="batch", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Vælg den klasse med højest score som vores forudsigelse
                predictions = output.logits.argmax(dim=-1)

                total_loss += output.loss.item()
                correct    += (predictions == labels).sum().item()
                total      += labels.size(0)

        return total_loss / len(loader), correct / total

    def predict(self, texts: list) -> list[str]:
        """
        Returnerer en liste af labels ("PII" eller "NON-PII") for nye tekster.
        """
        INV_LABEL_MAP = {0: "NON-PII", 1: "PII"}

        # Dummy labels da DataLoader kræver dem – de bruges ikke til noget her
        loader = self._make_dataloader(texts, ["NON-PII"] * len(texts), shuffle=False)

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluerer", unit="batch", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                output      = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output.logits.argmax(dim=-1).cpu().tolist()

                all_predictions.extend([INV_LABEL_MAP[p] for p in predictions])

        return all_predictions

    def print_evaluation_report(self, X_test: list, y_test: list):
        """
        Printer en detaljeret rapport med precision, recall og F1
        for hvert label – vigtigere end accuracy på ubalancerede data.
        """
        predictions = self.predict(X_test)
        print(classification_report(y_test, predictions, labels=["NON-PII", "PII"], zero_division=0))


# ─────────────────────────────────────────────
# 3. Pipeline
# ─────────────────────────────────────────────

class BertPipeline:
    """
    Sætter det hele sammen i én arbejdsgang.

    Den håndterer:
        1. Indlæsning og oprydning af data
        2. Opdeling i træning, validering og test
        3. Træning af BERT
        4. Evaluering og predictions
    """

    def __init__(self):
        self.trainer = BertTrainer()

    def preprocess(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Renser dataen og returnerer tekster og labels som lister.

        privacy-kolonnen indeholder numpy arrays med rå PII-annotations.
        Vi konverterer dem til simple "PII" / "NON-PII" strenge:
            - Er arrayet ikke tomt  --> teksten indeholder PII
            - Er arrayet tomt       --> teksten indeholder ingen PII
        """
        df = df.dropna(subset=["source_text"])

        texts = df["source_text"].tolist()

        # Konverter hvert privacy-array til "PII" eller "NON-PII"
        labels = ["PII" if len(p) > 0 else "NON-PII" for p in df["privacy"]]

        return texts, labels

    def run(self, df: pd.DataFrame) -> list:
        """
        Kører hele pipeline fra rå data til predictions.

        Opdelingen er:
            70% træning
            15% validering  (bruges til at følge med undervejs)
            15% test        (bruges kun til endelig evaluering)
        """
        texts, labels = self.preprocess(df)

        # Første split: træning vs. resten
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )

        # Andet split: validering vs. test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"Træning: {len(X_train)} eksempler")
        print(f"Validering: {len(X_val)} eksempler")
        print(f"Test: {len(X_test)} eksempler")

        # Træn modellen
        self.trainer.train(X_train, y_train, X_val, y_val)

        # Evaluer på testdata og print rapport
        print("\nEvaluering på testdata:")
        self.trainer.print_evaluation_report(X_test, y_test)

        # Returnér predictions
        return self.trainer.predict(X_test)


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

def main():
    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"

    try:
        df = pd.read_parquet(file_path)
        df = df.sample(2000, random_state=42)  # Skift til højere tal når du er klar til fuld træning

        pipeline    = BertPipeline()
        predictions = pipeline.run(df)

        print("\nEksempel på predictions:", predictions[:10])

    except Exception as e:
        print("Der opstod en fejl:", e)


if __name__ == "__main__":
    main()