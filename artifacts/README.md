# ExplainMyModel 🚀  
**Automated ML Audit, Risk & Explainability Engine**

ExplainMyModel is a production-grade machine learning audit system designed to evaluate models beyond accuracy — including business risk, data drift, calibration, bias, and interpretability.

It is built as a **modular engine** that works with any tabular dataset and supports configurable models, making it suitable for real-world ML governance and decision-making workflows.

---

## 🔍 What This System Does

Given a CSV dataset and a target column, ExplainMyModel will:

- Train a selected ML model
- Evaluate performance (accuracy, precision, recall, F1)
- Analyze class imbalance
- Estimate business cost (false positives / false negatives)
- Detect data drift
- Generate feature importance via SHAP
- Produce a structured audit payload (`payload.json`)
- (Optional) Generate a human-readable audit report using a local LLM

---

## 🏗 Project Structure

