
import pandas as pd
from typing import Tuple, List
import re

class Pipline:

    """ Klassen håndterer: Data preprocessing, Model loading,
    Træning og prediction for modeller: svm, regex, BERT.
    En generisk pipline til tekstklassifikation. """

    def __init__(self, model_type="svm"):

        """ Initialiserer pipeline med valgte model.
        Arsg: model_type(str) for modeller svm, regex, bert. """

        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):

        """ Oploader den valgte model baseret på model_type. """

        if self.model_type == 'svm':
            from sklearn.svm import SVC
            return SVC()

        elif self.model_type == 'regex':
            return None    # Regex kræver ikke træning

        elif self.model_type == 'bert':
            return "bert_model_pladceholder"    # Her kan man senere indsætte transformers

        else:
            raise ValueError(f"Ukendt model_type: {self.model_type}")

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:

        """ Forbereder data til modellen: --> 1. Fjernes rækker med manglende tekst. 2.Splitter data i features 
        (X), labels(y). Args: df (pd.DataFrame): Input dataframe --> Returns: Tuple[pd.Series, pd.Series]:(X, y). """

        df = df.dropna(subset=["source_text"])
        
        X = df["source_text"]
        y = df["privacy"]     # Labels-PII tags
        
        return X, y

    def run(self, df: pd.DataFrame) -> list:

        """ Kører hele pipline: preprocessing, træning(hvis nødvendigt), prediction.
        Arsg: pd.DataFrame) : input data --> Returener List: Prediction """

        X, y = self.preprocess(df)

        if self.model_type == "regex":
            return self.regex_predict(X)

        elif self.model_type == "svm":
            
            self.model.fit(X, y)
            
            return self.model.predict(X)

        elif self.model_type == "bert":
                
            # Her skal man senere indsætte rigtige infrence(Pladseholder)
            return ["BERT_predictions"] * len(X)

    def regex_predict(self, X: pd.Series) -> list[str]:

        """ Simpel regex_baseret PII detektion --> Ex: Finder e-mail eller CPR.
        Arsg: X:pd.Series: Tekstdata --> Retuner list[str] labels. """
                
        results = []

        for text in X:
            if re.search(r"\d{4}", text):
                results.append("PII")
            else:
                results.append("NON-PII")
        return results


def load_data(file_path: str) -> pd.DataFrame:

    """ Indlæser datasæt fra parquet-fil.
    file_path(str): sti til data --> Retuner Dataframe med data. """
    
    return pd.read_parquet(file_path)


def main():

    """ Indgangspunkt for script..
    - Loader data
    - Initialiserer pipeline
    - Kører modellen. """

    file_path = "hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet"

    try:
     
        df = load_data(file_path)    

        pipeline = Pipline(model_type="regex")    # Model typer vælges her svm/regex/tf-idf/bert

        predictions = pipeline.run(df)           

        print(predictions[:10])    # Vis eksempel 

    except Exception as e:
        print("Der opstod en fejl:", e)


if __name__ == "__main__":
    main()