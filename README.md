

# 🏡 California Housing Price Predictor

An **end-to-end machine learning project** demonstrating **data preprocessing, feature engineering, and model tuning** with Scikit-Learn pipelines.  
This project predicts **median house values in California districts** based on census data.

---

## 📌 Core Concepts Demonstrated

- ✅ **End-to-End Workflow** – From data fetching and cleaning to final model evaluation.  
- ✅ **Advanced Preprocessing** – Robust data cleaning, handling missing values, and transforming features.  
- ✅ **Scikit-Learn Pipelines** – Building a single, streamlined pipeline for all data transformations.  
- ✅ **Custom Transformers** – Engineering complex features like geographical cluster similarity.  
- ✅ **Hyperparameter Tuning** – Using `GridSearchCV` and `RandomizedSearchCV` to find the best model configuration.  
- ✅ **Model Evaluation** – In-depth analysis using cross-validation and final testing.  

---

## 🏗️ Project Workflow

```

```
  ┌─────────────────┐
  │  Data Ingestion │
  │   (housing.csv) │
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ Data Exploration│
  │   (EDA & Vis)   │
  └────────┬────────┘
           ▼
```

┌────────────────────────────┐
│  Preprocessing Pipeline    │
│ (Imputation, Scaling, OHE) │
└────────────┬───────────────┘
▼
┌────────────────────────────┐
│ Model Training & Tuning    │
│(Random Forest, GridSearchCV)│
└────────────┬───────────────┘
▼
┌──────────────────┐
│ Final Evaluation │
│   (on Test Set)  │
└──────────────────┘

````

---

## ⚙️ Tech Stack

- **Data Science:** Python, Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib  
- **Development Environment:** Jupyter Notebook  

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- Python 3.5+  
- An environment manager like **venv** or **conda**  

### 2️⃣ Setup & Launch
```bash
# Clone the repository
git clone https://github.com/your-username/california-housing-project.git
cd california-housing-project

# Create and activate a virtual environment (recommended)
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
````

* Open `end_to_end_project.ipynb` from your browser and run the cells to reproduce the workflow.

---

## 📈 Model Performance & Key Findings

* **Final Model:** Random Forest Regressor
* **Performance Metric:** Root Mean Squared Error (RMSE)
* **Final RMSE on Test Set:** $47,730

The final model's predictions are, on average, off by approximately **$47,730**.

**Most predictive features:**

* Median Income
* Geographical Location (via custom cluster similarity)
* Population Per Household

---

```

I can also create a **badge-style version with Docker, Python, Scikit-learn icons, and interactive visuals** like your DriftRadar README if you want it to look even more “professional GitHub ready.”  

Do you want me to do that next?
```
