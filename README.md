# AI Heart Risk Analyzer

An end-to-end machine learning project for predicting heart disease risk using clinical data, with an interactive Streamlit dashboard for real-time analysis.

---

## Features

* Real-time heart disease risk prediction
* Probability-based risk visualization
* Model-driven interpretation
* Interactive Streamlit interface
* Feature importance visualization

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## Project Structure

heart-disease-prediction/
│── app.py
│── requirements.txt
│── .gitignore

├── model/
│   ├── model.pkl
│   └── encoders.pkl

├── src/
│   ├── train.py
│   ├── preprocess.py
│   └── tester.py

├── notebook/
│   └── EDA.ipynb

├── data/
│   └── heart.csv

---

## How to Run

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

---


## Dataset

* File: heart.csv
* Source: UCI Heart Disease Dataset
* Included for reproducibility

---

## Model Details

* Algorithm: (mention your model, e.g. Random Forest)
* Task: Binary classification
* Output: Risk prediction with probability score

---

## Disclaimer

This project is for educational purposes only and not a substitute for medical advice.

---

## Author

Vasu Sharma
