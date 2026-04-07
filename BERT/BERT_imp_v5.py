# pip install transformers torch tqdm

import re
import time
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


dataset_size  = 266000
epochs        = 5
batch_size    = 96
learning_rate = 2e-5
max_len       = 128
save_path     = "saved_model_filtered"
device        = "cuda"

label_map = {
    "O": 0,
    "API_KEY": 1,
    "CREDIT_CARD_NUMBER": 2,
    "BANK_ACCOUNT_NUMBER": 3,
    "IBAN": 4,
    "PASSWORD": 5,
    "PASSPORT_NUMBER": 6,
    "SSN": 7,
    "FULL_NAME": 8,
    "FIRST_NAME": 9,
    "LAST_NAME": 10,
    "EMAIL": 11,
    "PHONE_NUMBER": 12,
}

inv_label_map = {v: k for k, v in label_map.items()}


def filter_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fjerner tekster der ligner struktureret data (JSON, CSV, logs).
    Baseret på to kriterier:

    1. Regex-detektion: teksten indeholder JSON-objekter, arrays
       eller key-value par som {"key": "value"} eller [item].

    2. Tegndensitet: mere end 5% af tegnene er strukturelle
       specialtegn som {, }, [, ], :, ;, |, ".
       Naturlig tekst har typisk under 2% sådanne tegn.

    Disse tekster kan skade modellen fordi BERT risikerer at lære
    at genkende PII via strukturelle markører (kolon, anførselstegn)
    frem for sproglig kontekst – præcis det vi vil undgå.
    """
    regex_pattern = re.compile(r'\{.*\}|\[.*\]|".*"\s*:', re.DOTALL)
    special_chars = re.compile(r'[\{\}\[\]":;\|]')

    def is_structured(text: str) -> bool:
        if regex_pattern.search(text):
            return True
        density = len(special_chars.findall(text)) / max(len(text), 1)
        return density > 0.05

    mask         = df["source_text"].apply(is_structured)
    n_removed    = mask.sum()
    n_total      = len(df)

    print(f"Struktureret data fjernet: {n_removed:,} rækker "
          f"({n_removed / n_total * 100:.1f}% af datasættet)")

    return df[~mask].copy()


class PIIDataset(Dataset):
    """
    Forbereder tekst og token-niveau labels til BertForTokenClassification.

    For hvert eksempel:
      - Tokenizerer teksten og returnerer input_ids og attention_mask
      - Bruger offset mapping til at finde hvilke tokens der dækker
        hver PII-entitet, og sætter det tilsvarende label
      - Tokens uden PII får label 0 (O)
    """

    def __init__(self, texts: list, entities: list, tokenizer: BertTokenizer, max_len: int = 128):
        self.texts     = texts
        self.entities  = entities
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text          = self.texts[idx]
        text_entities = self.entities[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)
        labels         = torch.zeros(self.max_len, dtype=torch.long)

        for entity in text_entities:
            label_name  = entity.get("label", "O")
            entity_text = entity.get("value", "")

            if label_name not in label_map:
                continue

            label_id   = label_map[label_name]
            start_char = text.find(entity_text)
            if start_char == -1:
                continue
            end_char = start_char + len(entity_text)

            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                token_start = token_start.item()
                token_end   = token_end.item()
                if token_start == 0 and token_end == 0:
                    continue
                if token_start >= start_char and token_end <= end_char:
                    labels[token_idx] = label_id

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


class BertTrainer:

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = learning_rate
        self.max_len    = max_len
        self.device     = device

        print(f"Bruger device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model     = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_map)
        ).to(self.device)

        self.use_amp = self.device == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # O-klassen dominerer (~90% af tokens) og får lav vægt så
        # modellen ikke blot lærer at gætte O på alt.
        class_weights = torch.ones(len(label_map), device=self.device)
        class_weights[label_map["O"]]               = 0.3
        class_weights[label_map["PASSPORT_NUMBER"]] = 1.5
        class_weights[label_map["FULL_NAME"]]       = 1.5
        class_weights[label_map["LAST_NAME"]]       = 1.5
        class_weights[label_map["FIRST_NAME"]]      = 1.2

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def _make_dataloader(self, texts: list, entities: list, shuffle: bool) -> DataLoader:
        dataset = PIIDataset(texts, entities, self.tokenizer, self.max_len)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    def train(self, X_train: list, y_train: list, X_val: list, y_val: list):
        """
        Fine-tuner BERT med early stopping baseret på val loss.
        Gemmer den bedste model efter hvert epoch og genindlæser
        den hvis val loss begynder at stige.
        """
        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader   = self._make_dataloader(X_val,   y_val,   shuffle=False)

        optimizer   = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")
        best_epoch    = 1

        for epoch in range(1, self.epochs + 1):
            train_loss        = self._train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self._evaluate(val_loader)

            print(f"Epoch {epoch}/{self.epochs}  |  "
                  f"Train loss: {train_loss:.4f}  |  "
                  f"Val loss: {val_loss:.4f}  |  "
                  f"Val accuracy: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                self.save(save_path)
                print(f"  Ny bedste model gemt (val loss: {val_loss:.4f})")
            else:
                print(f"  Val loss steg – stopper tidligt efter epoch {epoch}")
                print(f"  Indlæser bedste model fra epoch {best_epoch} "
                      f"(val loss: {best_val_loss:.4f})")
                self.load(save_path)
                break

    def _train_one_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc="  Træner", unit="batch", leave=False)
        for batch in progress:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = self.loss_fn(
                output.logits.view(-1, len(label_map)),
                labels.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluerer", unit="batch", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss   = self.loss_fn(
                    output.logits.view(-1, len(label_map)),
                    labels.view(-1)
                )

                predictions   = output.logits.argmax(dim=-1)
                active_tokens = attention_mask.bool()

                total_loss += loss.item()
                correct    += (predictions[active_tokens] == labels[active_tokens]).sum().item()
                total      += active_tokens.sum().item()

        return total_loss / len(loader), correct / total

    def print_evaluation_report(self, X_test: list, y_test: list):
        loader = self._make_dataloader(X_test, y_test, shuffle=False)
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                output      = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output.logits.argmax(dim=-1)
                active      = attention_mask.bool()

                all_preds.extend(predictions[active].cpu().tolist())
                all_labels.extend(labels[active].cpu().tolist())

        print(classification_report(
            all_labels, all_preds,
            labels=list(label_map.values()),
            target_names=list(label_map.keys()),
            zero_division=0,
        ))

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model gemt til: {path}")

    def load(self, path: str):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model     = BertForTokenClassification.from_pretrained(path).to(self.device)
        print(f"Model indlæst fra: {path}")

    def predict_sentence(self, text: str):
        """Hurtig test – printer PII fundet i én sætning."""
        dummy_loader = self._make_dataloader([text], [[]], shuffle=False)
        self.model.eval()

        with torch.no_grad():
            batch       = next(iter(dummy_loader))
            input_ids   = batch["input_ids"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            output      = self.model(input_ids=input_ids, attention_mask=attn_mask)
            predictions = output.logits.argmax(dim=-1).cpu()[0]

        tokens    = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        pii_found = []

        for token, pred, active in zip(tokens, predictions, attn_mask[0].cpu()):
            if not active or token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            label = inv_label_map[pred.item()]
            if token.startswith("##") and pii_found:
                pii_found[-1]["token"] += token[2:]
            else:
                pii_found.append({"token": token, "label": label})

        pii_found = [t for t in pii_found if t["label"] != "O"]

        if not pii_found:
            print("Ingen PII fundet.")
        else:
            print(f"\nTekst: '{text}'\n" + "-" * 40)
            for t in pii_found:
                print(f"  {t['token']:<25} → {t['label']}")


class BertPipeline:

    def __init__(self):
        self.trainer = BertTrainer()

    def preprocess(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Renser data og returnerer tekster og entiteter.

        Filtrerer struktureret data (JSON/CSV/logs) fra før træning
        så modellen lærer kontekstuel PII-genkendelse frem for at
        lære strukturelle mønstre som kolon og anførselstegn.
        """
        df = df.dropna(subset=["source_text"])
        df = filter_structured_data(df)

        texts    = df["source_text"].tolist()
        entities = [list(p) for p in df["privacy"]]

        return texts, entities

    def run(self, df: pd.DataFrame):
        texts, entities = self.preprocess(df)

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, entities, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"Træning:    {len(X_train)} eksempler")
        print(f"Validering: {len(X_val)} eksempler")
        print(f"Test:       {len(X_test)} eksempler")

        start   = time.time()
        self.trainer.train(X_train, y_train, X_val, y_val)
        elapsed = time.time() - start

        print(f"\nTræning færdig – tog {int(elapsed // 60)} min {int(elapsed % 60)} sek")
        print("\nEvaluering på testdata:")
        self.trainer.print_evaluation_report(X_test, y_test)

        return self.trainer.predict_sentence("Ring til Jonas Hansen på Jonas@gmail.com")


def main():
    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"

    try:
        df = pd.read_parquet(file_path)
        df = df.sample(dataset_size, random_state=42)

        pipeline = BertPipeline()
        pipeline.run(df)
        pipeline.trainer.save(save_path)

    except Exception as e:
        print("Der opstod en fejl:", e)


if __name__ == "__main__":
    main()