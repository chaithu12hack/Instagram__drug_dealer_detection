import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from utils.preprocess import load_and_preprocess


X, y = load_and_preprocess("dataset/posts.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "svm": SVC(probability=True)
}

os.makedirs("ml_models/models", exist_ok=True)

for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", model)
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"ml_models/models/{name}.pkl")
    print(f"[✔] Saved: {name}.pkl")
    print(f"[✔] Accuracy: {pipeline.score(X_test, y_test) * 100:.2f}%")
    print(f"[✔] Classification Report:\n{pipeline.score(X_test, y_test)}\n")