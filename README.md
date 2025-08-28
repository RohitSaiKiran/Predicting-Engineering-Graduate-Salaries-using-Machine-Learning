# Predicting-Engineering-Graduate-Salaries-using-Machine-Learning

An end-to-end **machine learning pipeline** that forecasts **engineering graduates’ salaries** using academic records, aptitude scores, domain expertise, and personality attributes.  
This repository is structured as a **hands-on learning kit**, showcasing how to take a raw dataset, clean it, explore it, prepare features, and build predictive regression models in a transparent and reproducible way.

## 🎯 Motivation

While many salary prediction examples jump straight to advanced models, they often neglect **clarity in preprocessing** or **baseline comparisons**.  
This project was designed to be a **clear and practical reference** by:

- Demonstrating a **stepwise workflow** from raw data → insights → models.
- Comparing **interpretable baselines** with **regularized regressors**.
- Including a **robust validation scheme** to ensure results aren’t misleading.
- Providing a **portable notebook** that works seamlessly if you simply drop your CSV into a `Data/` folder.

---

## ✨ What Makes It Special

- **Complete lifecycle**: EDA → cleaning → scaling/encoding → splitting → model training → evaluation.
- **Automatic dataset handling**: Notebook intelligently searches for the dataset under `Data/`.
- **Baseline first**: Linear Regression is used as a ground truth for performance comparisons.
- **Regularization focus**: Ridge & Lasso with cross-validation to prevent overfitting.
- **Lightweight feature selection demo**: Correlation-driven feature ranking for interpretability.

---

## 📖 Notebook Roadmap

- **Section 1:** Import dependencies & environment setup
- **Section 2:** Dataset discovery & loading (auto path detection)
- **Section 3:** Initial checks (shape, dtypes, missing values)
- **Section 4:** Exploratory Data Analysis (distribution plots, correlations)
- **Section 5:** Feature engineering
  - Normalization with **MinMaxScaler**
  - One-hot encoding for categorical variables
- **Section 6:** Model development
  - Train/Val/Test split (60/20/20 strategy)
  - **Linear Regression** baseline
  - **Ridge & Lasso Regression** with `GridSearchCV`
  - **Feature subset experiments** based on correlations
- **Section 7:** Evaluation
  - RMSE scores on validation and test sets
  - Comparison table of all models

---

## 🛠️ Tech Stack

- **Python Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **IDE/Tools:** Jupyter Notebook, VS Code
- **ML Focus:** Regression, feature preprocessing, model validation
- **Data Source:** Engineering Graduate Salary dataset (`Data/Engineering_graduate_salary.csv`)

---

## 📊 Dataset

The dataset used in this project is the **Engineering Graduate Salary Prediction** dataset, originally published on [Kaggle](https://www.kaggle.com/datasets/manishkc06/engineering-graduate-salary-prediction/data).  
For convenience and reproducibility, a copy (`Engineering_graduate_salary.csv`) is included under the `Data/` folder in this repository.

## 📂 Repository Layout

```
Graduate-Salary-Prediction-ML/
│── README.md
│── LICENSE
│── requirements.txt
│
├── notebooks/
│   └── Graduate-Salary-Prediction.ipynb
│
└── Data/
    └── Engineering_graduate_salary.csv
```

The notebook has **relative path detection** to make it runnable from repo root or directly from the `notebooks/` directory.

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/RohitSaiKiran/Predicting-Engineering-Graduate-Salaries-using-Machine-Learning
   ```

2. **Set up a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Place your dataset in the Data/ folder**

   ```
   Data/
   └── Engineering_graduate_salary.csv
   ```

4. **Run the notebook**
   ```
   notebooks/Graduate-Salary-Prediction.ipynb
   ```

---

## 🧩 Model Insights

- **Target variable:** Salary
- **Preprocessing:**
  - Scaled numeric fields (e.g., GPA, 10th/12th percentages, aptitude/domain scores).
  - Encoded categorical fields (e.g., gender, degree, specialization, boards).
- **Splitting strategy:** Two-step split → 60% train, 20% validation, 20% test.
- **Models implemented:**
  - Linear Regression (baseline)
  - Ridge Regression (grid search over alphas)
  - Lasso Regression (grid search over alphas)
- **Metric:** RMSE for validation and test sets.

---

## 📌 Next Steps

- Introduce **ElasticNet** for hybrid regularization.
- Experiment with **ensemble regressors** (Random Forest, Gradient Boosting).
- Build an **interactive dashboard** for visualizing salary prediction trends.
- Add experiment tracking with **MLflow**.

---

## 👤 Author

**Rohit Sai Kiran Ravula**  
📧 rohitsaikiran.r@gmail.com  
🔗 [GitHub](https://github.com/RohitSaiKiran)
