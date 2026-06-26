# 🇳🇬 Automated Nigerian News Verification Portal
### *An Engineering Framework for Multi-Classifier Ensemble Machine Learning Architectures*

---

## 🎯 Project Overview
I developed this automated news verification portal as the core empirical validation component of my Master's thesis methodology. The primary goal of this framework is to evaluate how effectively classical machine learning architectures, combined via a democratic ensemble layer, can detect and classify localized misinformation within the complex Nigerian political and socio-economic landscapes. 

Rather than relying on uncalibrated black-box deep learning models, I intentionally engineered a high-performance **Hard Voting Ensemble Network** to establish a reproducible, production-ready baseline benchmark using high-quality data curated directly from premium West African fact-checking organizations (Dubawa and FactCheckHub).

---

## 🔬 Core System Architecture
My verification pipeline is structured into three continuous computational layers:

### 1. Robust Linguistic Feature Engineering
To eliminate contextual noise and potential data leakage, I built a custom text-cleaning pipeline that filters raw text into a strict 5,000-dimensional sparse vocabulary matrix.
* **Normalization:** Handles lowercasing, strips raw URLs, and drops special character signatures.
* **Advanced Text Preprocessing:** Conducts standard `nltk` unigram/bigram tokenization followed by lemmatization to extract semantic base forms.
* **Target Leakage Mitigation:** Implements an explicit blacklist to filter out operational meta-tokens like `dubawa`, `factcheckhub`, `claims`, `image`, and `video`, preventing the models from learning artificial shortcuts.

### 2. The Hard Voting Ensemble Network
My backend engine orchestrates a consensus prediction across four distinct baseline sub-estimators. This structural plurality mitigates individual classifier variance and prevents localized overfitting on class-imbalanced regional datasets:
* **Multinomial Naive Bayes (MNB):** Captures foundational probabilistic word-frequency features.
* **Linear Support Vector Machine (Linear SVC):** Establishes optimized high-dimensional hyperplane separation boundaries.
* **Random Forest Classifier (RF):** Models non-linear combinations through bagged decision tree sub-spacing.
* **Gradient Boosting Classifier (GB):** Progressively minimizes remaining residual errors via sequential tree boosting.

### 3. Streamlit Validation Interface
I migrated the original frontend from a standard Flask/HTML micro-framework to an integrated, data-driven Streamlit architecture. This dashboard provides:
* **Live In-App Testing:** A dynamic lookup selector linked directly to my unseen `production_test_set.csv`.
* **Ground-Truth Tracking:** Displays the pristine, fact-checker-validated label alongside live model calculations.
* **Real-Time Engine Diagnostics:** Provides the exact voting consensus percentage split (e.g., 75% or 100%) and interactive bar charts illustrating model alignment vs. ground-truth divergence.

---

## 📦 Technical Stack & Environment Dependencies
The system utilizes specific, locked dependencies to preserve exact model unpickling configurations across environments:

* **Core Runtime:** `Python 3.11+`
* **Web UI Framework:** `Streamlit (v1.35.0)`
* **Machine Learning Pipelines:** `Scikit-Learn (v1.5.1)` & `Joblib (v1.4.2)`
* **Data Ingestion & Matrix Arrays:** `Pandas (v2.2.2)` & `NumPy (v1.26.4)`
* **Natural Language Processing:** `NLTK (v3.8.1)`

---

## 🚀 Local Installation & Deployment Guide

Follow these steps to set up and execute the verification web application on your local machine:

### 1. Clone the Workspace
Open your terminal and clone the repository using Git, then step into the project directory:
```bash
git clone [https://github.com/yourusername/nigerian-news-verification-ensemble.git](https://github.com/yourusername/nigerian-news-verification-ensemble.git)
cd nigerian-news-verification-ensemble
```

### 2. Setup and Activate Your Virtual Environment
Create an isolated environment to ensure package versions do not conflict with your global Python system setups:

* **For Windows (PowerShell/CMD):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

* **For macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Pinpoint Project Requirements
With your virtual environment active—indicated by `(.venv)` in your terminal prompt—install the pinned environment dependencies:
```bash
pip install -r requirements.txt
```

### 4. Boot Up the Streamlit Engine
Launch the web interface. The backend logic will handle missing background NLTK corpuses natively on the initial startup:
```bash
streamlit run streamlit_app.py
```
*The engine will spin up a local processing container and automatically mount the live, fully interactive dashboard at `http://localhost:8501` in your default browser.*

---

## 🎓 Academic Contribution
This repository moves beyond basic exploratory notebook modeling by showcasing an end-to-end, reproducible software engineering implementation of an academic methodology. It bridges the gap between text classification theory and interactive empirical validation, creating a viable prototype template for automated mis- and disinformation tracking in developing democratic regions.