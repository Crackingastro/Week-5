Here’s a fresh, emoji-packed `README.md` you can drop into your repo root:

# 🚀 Credit Risk Scoring – Week 5 Project

> A full end-to-end credit-risk scoring pipeline & REST API built with scikit-learn, MLflow, and FastAPI.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/fastapi-%5E0.115.14-green.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/mlflow-%5E3.1.1-orange.svg" alt="MLflow" />
  <img src="https://img.shields.io/badge/tests-pytest-red.svg" alt="pytest" />
</p>

## ✨ Features

- ✂️ **Data Preprocessing**  
  – Feature engineering (timestamp breakdown)  
  – Imputation, scaling & encoding  
- 🧠 **Model Training & Tracking** via MLflow  
- 🚀 **Live Prediction API** with FastAPI  
- 🧪 **Automated Tests** (pipeline & model)  
- 📓 **Jupyter Notebooks** for EDA & prototyping  

## 📦 Quick Start

1. **Clone**  
   ```bash
   git clone https://github.com/Crackingastro/Week-5.git
   cd Week-5


2. **Create & activate venv**

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   ```
3. **Install deps**

   ```bash
   pip install -r requirements.txt
   ```
4. **Fetch data** (DVC)

   ```bash
   dvc pull
   ```
5. **Run the API**

   ```bash
   uvicorn src.api.main:app --reload
   ```
6. **Hit the endpoint**

   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{ "TransactionId":"…", "BatchId":"…", …, "FraudResult":0 }'
   # → { "prediction": 0 }
   ```

## 🛠️ Project Structure

```
📦 Week-5
├── Data/                # raw & processed data (DVC-tracked)
├── notbooks/            # Jupyter notebooks (EDA, model training)
├── src/
│   ├── api/             # FastAPI app & Pydantic schemas
│   └── model/           # Pipeline & model pickles (+ builder script)
├── tests/               # pytest suite for pipeline & model
├── requirements.txt     # Python dependencies
└── readme.md            # ← you are here!
```

## 🛠️ Usage

### 1. Build & save preprocessing pipeline

```bash
python src/model/pipeline.py
```

### 2. (Re)train model

> 📓 Check `notbooks/` for the MLflow-powered training notebooks.

### 3. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 4. Make predictions

* **Endpoint:** `POST /predict`
* **Payload:** JSON with all 16 transaction fields
* **Response:** `{ "prediction": 0 | 1 }`

## 🧪 Testing

Run the full test suite:

```bash
pytest -v
```

* **test\_pipeline.py**: validates feature names & input checks
* **test\_model.py**: checks model loading & output shape

## 🤝 Contributing

1. ⭐ Star the repo
2. 🍴 Fork it
3. 📥 Create a feature branch
4. 🔀 Submit a pull request

---

Made with ❤️ by **Crackingastro**

```

Feel free to tweak any section or emoji to suit your style—enjoy!
```
