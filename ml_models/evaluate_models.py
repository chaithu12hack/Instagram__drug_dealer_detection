import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

from tensorflow.keras.models import load_model
from dl_model.train_lstm import X_test, y_test  # Assumes test data is exposed here

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Dict to store all metrics
all_metrics = {}

# Store ROC curves for combined plot
roc_curves = []

def evaluate_model(name, y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"results/confusion_matrix_{name.lower()}.png")
    plt.close()

    # Store ROC for combined plot
    roc_curves.append((name, fpr, tpr, auc_score))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc_score
    }

# === Evaluate LSTM ===
print("🔍 Evaluating LSTM...")
lstm_model = load_model("dl_model/lstm_model.h5")
lstm_probs = lstm_model.predict(X_test).flatten()
lstm_preds = (lstm_probs > 0.5).astype(int)
all_metrics["LSTM"] = evaluate_model("LSTM", y_test, lstm_preds, lstm_probs)

# === Evaluate ML Models ===
print("🔍 Evaluating ML models...")
for model_name in ["RandomForest", "SVM", "DecisionTree"]:
    model_path = f"ml_models/models/{model_name.lower()}.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Get prediction probabilities or decision function
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
            else:
                y_probs = model.decision_function(X_test)
                y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())  # normalize

            y_preds = model.predict(X_test)
            all_metrics[model_name] = evaluate_model(model_name, y_test, y_preds, y_probs)
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    else:
        print(f"⚠️ Skipped {model_name} — model file not found.")

# === Save metrics.json ===
with open("results/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

# === Accuracy Comparison Plot ===
plt.figure(figsize=(8, 6))
names = list(all_metrics.keys())
accs = [all_metrics[m]["accuracy"] for m in names]
plt.bar(names, accs, color="skyblue")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for i, acc in enumerate(accs):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
plt.savefig("results/accuracy_plot.png")
plt.close()

# === Combined ROC Curve Plot ===
plt.figure(figsize=(8, 6))
for name, fpr, tpr, auc_score in roc_curves:
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.savefig("results/roc_curve_all_models.png")
plt.close()

print("✅ All metrics and plots saved to results/")
