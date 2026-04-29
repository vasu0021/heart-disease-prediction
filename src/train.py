from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

print("🚀 Starting Training with Multiple Models...")

# Load data
df, encoders = load_and_preprocess("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("\n📊 Model Comparison:\n")

# Train & evaluate
for name, model in models.items():
    print(f"🔹 Training {name}...")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results[name] = (model, acc, auc)

    print(f"{name} → Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    print(classification_report(y_test, preds))
    print("-" * 50)

# 🔥 Find best model
best_model_name = max(results, key=lambda x: results[x][1])
best_model = results[best_model_name][0]

print(f"\n🏆 Best Model: {best_model_name}")

# Save best model
joblib.dump(best_model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("💾 Best model & encoders saved!")

# 📈 ROC Curve
plt.figure(figsize=(8,6))

for name, (model, acc, auc) in results.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()