
import pandas as pd
from typing import Tuple, List
<<<<<<< HEAD
import re
=======
from sklearn.model_selection import train_test_split
from tfidf_model import build_vectorizers, fit_transform_features
>>>>>>> d0e01ea9184d1e558419d7e86638ac58ce66216a

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
            
            # Lav train/val/test split
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            # Byg features med TF-IDF vectorizers
            X_train_features, X_val_features, X_test_features = fit_transform_features(X_train, X_val, X_test)
            
            # Træn modellen på de transformerede features
            self.model.fit(X_train_features, y_train)
            
            # Predictions på test data
            predictions = self.model.predict(X_test_features)
            
            return predictions

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