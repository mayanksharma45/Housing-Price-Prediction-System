# 🏠 Housing Price Prediction System

An end-to-end **Machine Learning project** that predicts housing prices based on multiple property features.
The project covers the complete ML lifecycle — data ingestion, preprocessing, model training, evaluation, and deployment using **Streamlit**.

---

## 🚀 Live Demo

👉 Deployed on **Streamlit Cloud**

> Users can input house details and get an estimated sale price instantly.

---

## 📌 Features

* End-to-end ML pipeline (training → inference)
* Robust data preprocessing with categorical & numerical handling
* Multiple regression models (including CatBoost / XGBoost)
* Interactive web UI built with Streamlit
* Production-safe preprocessing (`handle_unknown="ignore"`)
* Python 3.11 compatible & deployment-ready

---

## 🗂️ Project Structure

```
Housing-Price-Prediction-System/
│
├── artifacts/                  # Trained model & preprocessor (.pkl)
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── utils.py
│   ├── exception.py
│   └── logger.py
│
├── research/
│   ├── EDA_Houses_Prices.ipynb
│   └── Model_Training.ipynb
│
├── app.py                      # Streamlit application
├── main.py                     # Training pipeline entry point
├── requirements.txt
├── pyproject.toml
├── runtime.txt                 # Python version for Streamlit Cloud
└── README.md
```

---

## ⚙️ Tech Stack

* **Language:** Python 3.11
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * catboost, xgboost
  * streamlit
* **Deployment:** Streamlit Cloud

---

## 🧠 Machine Learning Workflow

1. **Data Ingestion**

   * Load and split raw housing data

2. **Data Transformation**

   * Numerical scaling
   * Categorical encoding using `OneHotEncoder(handle_unknown="ignore")`

3. **Model Training**

   * Multiple regression models trained and evaluated
   * Best-performing model selected

4. **Model Serialization**

   * Trained model and preprocessor saved as `.pkl` files

5. **Inference**

   * User input → preprocessing → prediction via Streamlit UI

---

## 🖥️ Running the Project Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mayanksharma45/Housing-Price-Prediction-System.git
cd Housing-Price-Prediction-System
```

---

### 2️⃣ Create virtual environment (recommended: `uv`)

```bash
uv venv --python 3.11
.venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
uv pip install -r requirements.txt
```

---

### 4️⃣ Train the model (optional if artifacts already exist)

```bash
python -m src.components.data_ingestion
python -m src.components.data_transformation
python -m src.components.model_trainer
```

---

### 5️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Cloud

* Python version specified via `runtime.txt`

  ```
  python-3.11
  ```
* Required dependencies listed in `requirements.txt`
* Model artifacts (`artifacts/model.pkl`, `artifacts/preprocessor.pkl`) must be:

  * committed to GitHub **OR**
  * downloaded at runtime from external storage

---

## ⚠️ Important Notes

* **Model artifacts are required at runtime** for prediction.
* Ensure training and inference environments use compatible versions:

  * Python 3.11
  * NumPy < 2.0
  * scikit-learn 1.3.2
* Categorical features are handled safely to avoid unseen-category errors.

---

## 📈 Future Improvements

* Single unified sklearn `Pipeline`
* Model versioning
* External model storage (S3 / GDrive / Hugging Face)
* Enhanced UI with validation & charts
* API version using FastAPI

---

## 👨‍💻 Author

**Mayank Sharma**

📌 GitHub: [https://github.com/mayanksharma45](https://github.com/mayanksharma45)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it really helps!