# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers (Setosa, Versicolor, Virginica) based on their sepal and petal measurements, built as part of the **CodeAlpha Data Science Internship**.

The project covers the full data science workflow: exploratory data analysis, model comparison, evaluation, and deployment through an interactive web app.

## 🎯 Project Overview

- **Task:** Multi-class classification of Iris species
- **Dataset:** Iris dataset (built-in via `scikit-learn`, 150 samples, 4 features, 3 balanced classes)
- **Final Model:** Logistic Regression
- **Test Accuracy:** 97%
- **Deployment:** Interactive Streamlit web app

## 🔍 Workflow

1. **Data Quality Check** — verified no missing values; found and documented 1 duplicate row (kept, as this is the standard, well-known Iris dataset)
2. **Exploratory Data Analysis** — analyzed distributions and correlations; found that `petal length` and `petal width` are the strongest features for separating species, while `sepal width` overlaps heavily across classes
3. **Train/Test Split** — 80/20 split with `stratify` to preserve class balance
4. **Cross-Validation** — used 5-fold CV to get a reliable performance estimate instead of relying on a single split
5. **Model Comparison** — compared Logistic Regression, KNN, Decision Tree, Random Forest, and SVM using cross-validation
6. **Final Evaluation** — confusion matrix and classification report on the held-out test set
7. **Deployment** — saved the trained model with `joblib` and built a Streamlit GUI for live predictions

## 📊 Model Comparison Results

| Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| **Logistic Regression** | **97.33%** | 2.49% |
| KNN | 97.33% | 2.49% |
| Random Forest | 96.67% | 2.11% |
| SVM | 96.67% | 2.11% |
| Decision Tree | 95.33% | 3.40% |

**Logistic Regression** was selected as the final model — it matched the top accuracy while being the most interpretable (clear feature coefficients).

## 🧪 Final Test Set Results

- **Accuracy:** 97%
- Setosa and Virginica: classified perfectly (10/10)
- Versicolor: 1 misclassification (predicted as Virginica) — consistent with the mild overlap observed during EDA

## 🖥️ Streamlit App

A simple web interface where you can adjust sliders for sepal/petal measurements and get an instant species prediction with confidence score.

### Run it locally

```bash
git clone https://github.com/menna240/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
