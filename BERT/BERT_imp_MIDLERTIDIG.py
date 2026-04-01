# Hvis kode skal køres skal der først installeres disse pakker: pip install transformers torch tqdm

import pandas as pd
from typing import Tuple
import re
import time
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm



# INDSTILLINGER – juster disse værdier for at tune modellen

# Antal eksempler der bruges til træning (max ~266000)
# Jo flere, jo bedre model – men jo længere tid tager det
dataset_size = 100000

# Antal gange modellen ser hele datasættet igennem
# For lidt = modellen lærer ikke nok, for mange = overfitting
epochs = 3

# Hvor mange eksempler modellen behandler ad gangen
# Større batch = hurtigere, men kræver mere VRAM
# RTX 3070 (8GB): 128 er fint med fp16
batch_size = 128

# Hvor hurtigt modellen opdaterer sine vægte
# 2e-5 er standard for BERT – rør den ikke medmindre du ved hvad du laver
learning_rate = 2e-5

# Maks antal tokens per tekst – tekster der er længere bliver klippet
# BERT understøtter maks 512
max_len = 128

# Hvor modellen gemmes efter træning
save_path = "saved_model"

# Hvilken enhed modellen kører på – "cuda" = GPU, "cpu" = CPU
device = "cuda"



 
# 1. PIIDataset
 

# De labels vi arbejder med – "O" betyder "ikke PII"
label_map = {
    "O":              0,
    "FULL_NAME":      1,
    "EMAIL":          2,
    "PHONE_NUMBER":   3,
    "STREET_ADDRESS": 4,
    "CITY":           5,
    "DATE":           6,
    }

# Bruges til at oversætte tal tilbage til label-navne
INV_label_map = {v: k for k, v in label_map.items()}


class PIIDataset(Dataset):
    """
    Forbereder tekst og labels til BERT token-klassifikation.

    Forskellen fra v1:
        I stedet for ét label per tekst, skal vi nu give ét label
        per token. Det kræver at vi finder ud af præcis hvilke
        tokens der svarer til PII-entiteterne i teksten.

    Eksempel:
        Tekst:  "Mit navn er Jonas Hansen"
        Labels: [ O,   O,   O,  FULL_NAME, FULL_NAME ]

    Args:
        texts    : Liste af rå inputtekster
        entities : Liste af lister med PII-entiteter per tekst
                   Hver entitet er en dict: {"label": "FULL_NAME", "value": "Jonas Hansen"}
        tokenizer: BERTs tokenizer
        max_len  : Maks antal tokens per tekst (default 128)
    """

    def __init__(self, texts: list, entities: list, tokenizer: BertTokenizer, max_len: int = 128):
        self.texts     = texts
        self.entities  = entities
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text            = self.texts[idx]
        text_entities   = self.entities[idx]

        # Tokenizer med return_offsets_mapping=True så vi ved
        # hvilken del af den originale tekst hvert token dækker
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,   # Giver os (start, slut) position per token
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # [(start, slut), ...]

        # Start med at sætte alle tokens til "O" (ikke PII)
        labels = torch.zeros(self.max_len, dtype=torch.long)

        # Gå igennem hver PII-entitet og find de tokens der dækker den
        for entity in text_entities:
            label_name  = entity.get("label", "O")
            entity_text = entity.get("value", "")

            # Spring over labels vi ikke arbejder med
            if label_name not in label_map:
                continue

            label_id = label_map[label_name]

            # Find hvor i teksten denne entitet starter og slutter
            start_char = text.find(entity_text)
            if start_char == -1:
                continue   # Entiteten findes ikke i teksten – spring over
            end_char = start_char + len(entity_text)

            # Gå igennem alle tokens og sæt label hvis tokenet
            # overlapper med entitetens position i teksten
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                token_start = token_start.item()
                token_end   = token_end.item()

                # Special tokens som [CLS] og [SEP] har offset (0, 0) – skip dem
                if token_start == 0 and token_end == 0:
                    continue

                # Tjek om tokenet overlapper med entiteten
                if token_start >= start_char and token_end <= end_char:
                    labels[token_idx] = label_id

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


 
# 2. BertTrainer
 

class BertTrainer:
    """
    Håndterer træning og evaluering af BERT til token-klassifikation.

    Forskellen fra v1:
        Modellen returnerer nu ét label per token i stedet for
        ét label per tekst. Evalueringen ignorerer padding-tokens
        da de ikke er en del af den rigtige tekst.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):

        self.model_name = model_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = learning_rate
        self.max_len    = max_len

        self.device = device
        print(f"Bruger device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # num_labels matcher antallet af labels i vores label_map (6 i alt)
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_map)
        ).to(self.device)

        # Mixed precision – bruger fp16 på GPU for dobbelt hastighed
        # På CPU er fp16 ikke understøttet, så vi deaktiverer det der
        self.use_amp = self.device == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _make_dataloader(self, texts: list, entities: list, shuffle: bool) -> DataLoader:
        dataset = PIIDataset(texts, entities, self.tokenizer, self.max_len)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,      # Loader data i baggrunden mens GPU'en træner
            pin_memory=True,    # Hurtigere dataoverførsel til GPU
        )

    def train(self, X_train: list, y_train: list, X_val: list, y_val: list):
        """
        Fine-tuner BERT på træningsdata og evaluerer efter hver epoch.
        """
        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader   = self._make_dataloader(X_val,   y_val,   shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        total_steps = len(train_loader) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, self.epochs + 1):
            train_loss           = self._train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc    = self._evaluate(val_loader)

            print(f"Epoch {epoch}/{self.epochs}  |  "
                  f"Train loss: {train_loss:.4f}  |  "
                  f"Val loss: {val_loss:.4f}  |  "
                  f"Val accuracy: {val_acc:.4f}")

    def _train_one_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc="  Træner", unit="batch", leave=False)
        for batch in progress:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += output.loss.item()
            progress.set_postfix(loss=f"{output.loss.item():.4f}")

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluerer", unit="batch", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Forudsigelse: vælg label med højest score per token
                predictions = output.logits.argmax(dim=-1)

                total_loss += output.loss.item()

                # Ignorer padding tokens (attention_mask == 0) i accuracy-beregningen
                # da de ikke er rigtige tokens og ville give et kunstigt højt resultat
                active_tokens = attention_mask.bool()
                correct += (predictions[active_tokens] == labels[active_tokens]).sum().item()
                total   += active_tokens.sum().item()

        return total_loss / len(loader), correct / total

    def predict(self, texts: list) -> list[list[dict]]:
        """
        Returnerer token-niveau predictions for en liste af tekster.

        Output eksempel for "Mit navn er Jonas Hansen":
            [
                {"token": "Mit",     "label": "O"},
                {"token": "navn",    "label": "O"},
                {"token": "er",      "label": "O"},
                {"token": "Jonas", "label": "FULL_NAME"},
                {"token": "Hansen",  "label": "FULL_NAME"},
            ]
        """
        # Dummy entities da DataLoader kræver dem
        dummy_entities = [[]] * len(texts)
        loader         = self._make_dataloader(texts, dummy_entities, shuffle=False)

        self.model.eval()
        all_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="  Forudsiger", unit="batch", leave=False)):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                output      = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output.logits.argmax(dim=-1).cpu()

                # Gå igennem hvert eksempel i batchen
                for i in range(input_ids.size(0)):
                    tokens      = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
                    token_preds = predictions[i]
                    mask        = attention_mask[i].cpu()

                    result = []
                    for token, pred, active in zip(tokens, token_preds, mask):
                        # Spring padding og special tokens over
                        if not active or token in ("[CLS]", "[SEP]", "[PAD]"):
                            continue
                        # Saml WordPiece-dele sammen igen
                    # BERT splitter f.eks. "hotmail" til "hot" + "##mail"
                    # Vi sætter dem sammen til ét token igen
                    if token.startswith("##") and result:
                        result[-1]["token"] += token[2:]
                    else:
                        result.append({
                            "token": token,
                            "label": INV_label_map[pred.item()]
                        })

                    all_results.append(result)

        return all_results

    def print_evaluation_report(self, X_test: list, y_test: list):
        """
        Printer precision, recall og F1 per label.
        Ignorerer O-labels da de udgør størstedelen af tokens og
        ville skævvride resultaterne.
        """
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

                # Kun aktive tokens
                active = attention_mask.bool()
                all_preds.extend(predictions[active].cpu().tolist())
                all_labels.extend(labels[active].cpu().tolist())

        label_names = list(label_map.keys())
        print(classification_report(
            all_labels, all_preds,
            labels=list(label_map.values()),
            target_names=label_names,
            zero_division=0
        ))


    def save(self, path: str):
        """Gemmer model og tokenizer til disk så du ikke skal træne forfra."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model gemt til: {path}")

    def load(self, path: str):
        """Indlæser en gemt model fra disk – overskriver den nuværende."""
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model     = BertForTokenClassification.from_pretrained(path).to(self.device)
        print(f"Model indlæst fra: {path}")

    def predict_sentence(self, text: str):
        """
        Test-funktion: skriv en sætning og se hvad modellen finder.

        Eksempel:
            trainer.predict_sentence("Ring til Jonas Hansen på 12345678")

        Output:
            Jonas    → FULL_NAME
            Hansen     → FULL_NAME
            12345678   → PHONE_NUMBER
        """
        results = self.predict([text])[0]
        pii_found = [t for t in results if t["label"] != "O"]

        if not pii_found:
            print("Ingen PII fundet i teksten.")
        else:
            print(f"\nTekst: '{text}'")
            print("-" * 40)
            for t in pii_found:
                print(f"  {t['token']:<25} → {t['label']}")

 
# 3. Pipeline
 

class BertPipeline:
    """
    Sætter det hele sammen i én arbejdsgang.

    Forskellen fra v1:
        preprocess() skal nu udtrække entiteterne fra privacy-kolonnen
        som en liste af dicts per tekst, i stedet for blot "PII"/"NON-PII".
    """

    def __init__(self):
        self.trainer = BertTrainer()

    def preprocess(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Renser data og returnerer tekster og entiteter.

        privacy-kolonnen indeholder numpy arrays af dicts, f.eks.:
            [{"label": "FULL_NAME", "value": "Jonas Hansen"}, ...]

        Vi konverterer dem til lister af dicts som PIIDataset forventer.
        """
        df = df.dropna(subset=["source_text"])

        texts    = df["source_text"].tolist()

        # Konverter hvert numpy array til en python liste
        entities = [list(p) for p in df["privacy"]]

        return texts, entities

    def run(self, df: pd.DataFrame) -> list:
        """
        Kører hele pipeline fra rå data til token-niveau predictions.
        """
        texts, entities = self.preprocess(df)

        # 70% træning, 15% validering, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, entities, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"Træning:    {len(X_train)} eksempler")
        print(f"Validering: {len(X_val)} eksempler")
        print(f"Test:       {len(X_test)} eksempler")

        start_time = time.time()
        self.trainer.train(X_train, y_train, X_val, y_val)
        elapsed = time.time() - start_time

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\nTræning færdig – tog {minutes} min {seconds} sek")

        print("\nEvaluering på testdata:")
        self.trainer.print_evaluation_report(X_test, y_test)

        return self.trainer.predict(X_test)


 
# 4. Main
 

def main():
    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"

    try:
        df = pd.read_parquet(file_path)
        df = df.sample(dataset_size, random_state=42)

        pipeline    = BertPipeline()
        predictions = pipeline.run(df)

        # Find den første tekst der faktisk indeholder PII og vis den
        # Gem modellen så du ikke skal træne forfra næste gang
        pipeline.trainer.save(save_path)

        # Test modellen med en vilkårlig sætning
        print("\nTest med en sætning:")
        pipeline.trainer.predict_sentence("Ring til Jonas Hansen på Jonas@gmail.com")

    except Exception as e:
        print("Der opstod en fejl:", e)


if __name__ == "__main__":
    main()


# HVAD GØR DENNE PIPELINE EGENTLIG?
#
# Målet er at finde persondata (PII) i fri tekst – ikke bare svare
# "ja der er PII i denne tekst", men præcis HVILKET ord der er PII
# og HVILKEN kategori det tilhører.
#
# Eksempel:
#   Input:  "Ring til Jonas Hansen på Jonas@gmail.com"
#   Output: "Jonas"          → FULL_NAME
#           "Hansen"           → FULL_NAME
#           "Jonas@gmail.com"→ EMAIL
#
  
# TRIN 1 – PIIDataset
  
# BERT kan ikke læse tekst direkte – den arbejder med tal.
# PIIDataset oversætter derfor hver tekst til tre ting:
#
#   input_ids      – hvert ord/token som et tal i BERTs vokabular
#   attention_mask – fortæller BERT hvilke tokens der er rigtige
#                    og hvilke der bare er tom padding
#   labels         – ét label per token, f.eks. [0, 0, 0, 1, 1, 2]
#                    hvor 0=O (ikke PII), 1=FULL_NAME, 2=EMAIL osv.
#
# For at finde ud af hvilke tokens der svarer til et PII-ord bruges
# offset mapping – det giver koordinater for hvert token i den
# originale tekst, så vi kan matche dem mod vores annotations.
#
  
# TRIN 2 – BertTrainer
# Vi bruger en færdigtrænret BERT model (bert-base-uncased) som
# allerede har lært sprogets struktur fra enorme mængder tekst.
# Ovenpå den sætter vi et klassifikationslag der lærer at bruge
# BERTs sprogforståelse til at genkende PII.
#
# Denne proces kaldes fine-tuning og er meget mere effektiv end
# at træne en model fra bunden, fordi BERT allerede ved hvad
# navne, emails og adresser er – vi skal bare lære den at pege på dem.
#
# Træningen kører i epochs (gennemløb af hele datasættet).
# Efter hvert epoch evaluerer vi på valideringsdata for at se
# om modellen faktisk bliver bedre – eller begynder at overfit.
#
# TRIN 3 – BertPipeline
# Pipeline er det der binder det hele sammen:
#
#   1. Indlæs og rens data fra datasættet
#   2. Split i træning (70%), validering (15%) og test (15%)
#   3. Træn modellen på træningsdata
#   4. Evaluer på testdata og print precision/recall/F1 per label
#   5. Returner predictions – hvad modellen mener hvert token er
#
# Evalueringen bruger F1-score frem for accuracy fordi datasættet
# er meget ubalanceret – langt de fleste tokens er O (ikke PII).
# En model der altid gætter O ville få høj accuracy men fange
# ingen PII overhovedet, hvilket er ubrugeligt.
#
# ═══════════════════════════════════════════════════════════════════