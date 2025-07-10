# Model Performance Report – Diabetes Prediction

## Model Used
**Support Vector Machine (SVM)** with linear kernel

---

## Dataset Summary
- **Source:** Pima Indians Diabetes Dataset
- **Total Samples:** 768
- **Features:** 8
- **Target:** Binary classification (0 = Non-diabetic, 1 = Diabetic)
- **Preprocessing:** StandardScaler applied to all features

---

## Train-Test Split
- **Train Size:** 80% (614 samples)
- **Test Size:** 20% (154 samples)
- **Stratification:** Yes
- **Random State:** 2

---

## Performance Metrics

### Training Set
- **Accuracy:** `0.7832`

### Test Set
- **Accuracy:** `0.7727`

---

## Observations
- The model performs consistently across training and test sets, indicating low variance and minimal overfitting.
- Linear SVM provides a solid baseline due to its speed and interpretability.

---

## Next Steps
- Compare results with other models (e.g., Logistic Regression, Random Forest, KNN)
- Evaluate with more metrics:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC-AUC Curve
- Address possible data imbalance using techniques like **SMOTE**
- Explore feature importance or SHAP analysis
