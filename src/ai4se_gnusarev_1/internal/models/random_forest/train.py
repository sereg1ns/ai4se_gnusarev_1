import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import yaml
import polars as pl


def main(
    cfg_path: str,
    save_path: str,
) -> RandomForestClassifier:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    train = pl.read_excel(cfg["data"]["train_path"])
    X_train, y_train = (
        train.get_column("message"),
        train.get_column("is_toxic"),
    )
    test = pl.read_excel(cfg["data"]["test_path"])
    X_test, y_test = test.get_column("message"), test.get_column("is_toxic")

    # Initialize TfidfVectorizer and RandomForestClassifier
    vectorizer = TfidfVectorizer()
    clf = RandomForestClassifier(**cfg["parameters"])

    # Fit vectorizer on training data and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train Random Forest classifier
    clf.fit(X_train_tfidf, y_train)

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict on test set
    test_preds = clf.predict(X_test_tfidf)

    # Calculate test metrics
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(
        y_test, test_preds, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, test_preds, average="weighted", zero_division=0
    )
    f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    # Log test metrics
    print(f"test Accuracy: {accuracy:.4f}")
    print(f"test Precision: {precision:.4f}")
    print(f"test Recall: {recall:.4f}")
    print(f"test F1 Score: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # save model
    model_name = os.path.split(os.path.split(cfg_path)[0])[1]
    save_model(clf, os.path.join(save_path, model_name))

    return accuracy


def save_model(model, path: str):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(model, f)
