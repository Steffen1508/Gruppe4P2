import joblib
from scipy.sparse import hstack

MODEL_DIR = "saved_model"

model = joblib.load(f"{MODEL_DIR}/model.joblib")
char_vectorizer = joblib.load(f"{MODEL_DIR}/char_vectorizer.joblib")
word_vectorizer = joblib.load(f"{MODEL_DIR}/word_vectorizer.joblib")
mlb = joblib.load(f"{MODEL_DIR}/mlb.joblib")
best_params = joblib.load(f"{MODEL_DIR}/best_params.joblib")


def predict_pii(text):
    X_char = char_vectorizer.transform([text])
    X_word = word_vectorizer.transform([text])

    X_all = hstack([X_char, X_word]).tocsr()

    y_pred = model.predict(X_all)

    labels = mlb.inverse_transform(y_pred)[0]

    return list(labels)


if __name__ == "__main__":
    text = "My email is valdemar@gmail.com and my phone number is +45 12345678"

    prediction = predict_pii(text)

    print("Text:")
    print(text)
    print("\nPredicted labels:")
    print(prediction)
    print("\nBest params:")
    print(best_params)