import joblib

enc = joblib.load("encoders.pkl")

print(enc['fbs'].classes_)
print(enc['exang'].classes_)