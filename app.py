import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- PAGE INITIALIZATION & CONFIGURATION ---
st.set_page_config(
    page_title="Nigerian News Verification Portal",
    page_icon="🇳🇬",
    layout="centered"
)

# Custom CSS injection for thesis branding alignment (Matches your previous Tailwind aesthetic)
st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; font-weight: 800; color: #1e1b4b; text-align: center; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #4338ca; text-align: center; margin-bottom: 2rem; font-weight: 500; }
        .verdict-box { padding: 1.5rem; border-radius: 0.75rem; text-align: center; font-weight: 900; font-size: 2rem; margin-top: 1rem; }
        .verdict-false { background-color: #fef2f2; color: #be123c; border: 1px solid #fecdd3; }
        .verdict-misleading { background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
        .verdict-true { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .metadata-panel { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 0.5rem; }
        footer-text { font-size: 0.75rem; color: #94a3b8; text-align: center; display: block; margin-top: 3rem; }
    </style>
""", unsafe_allow_html=True)


# Cache heavy system file IO assets to maximize memory efficiency
@st.cache_resource
def load_nlp_pipeline():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return WordNetLemmatizer(), set(stopwords.words('english'))


@st.cache_resource
def load_ml_engine():
    assets_dir = "web_system_assets"
    vec = joblib.load(os.path.join(assets_dir, "production_tfidf_vectorizer.joblib"))
    ens = joblib.load(os.path.join(assets_dir, "production_voting_ensemble.joblib"))
    enc = joblib.load(os.path.join(assets_dir, "target_label_encoder.joblib"))
    return vec, ens, enc


@st.cache_data
def load_evaluation_dataset():
    csv_path = "production_test_set.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# Load underlying architectural layers
lemmatizer, stop_words = load_nlp_pipeline()
vectorizer, ensemble_model, label_encoder = load_ml_engine()
test_df = load_evaluation_dataset()


def production_text_cleaner(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2])


# --- USER INTERFACE PRESENTATION LAYOUT ---
st.markdown("<div class='main-header'>Nigerian News Verification Portal</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Automated Misinformation Detection Using Multi-Classifier Ensemble Architectures</div>",
    unsafe_allow_html=True)

st.subheader("Step 1: Select or Paste a Claim")

# Initialize default empty session parameters
selected_headline = ""
ground_truth_label = None

if test_df is not None:
    # Build dropdown selections combining manual entry capabilities
    options = ["-- Select an Unseen Headline from Test Dataset --"] + test_df[
        'news_headline'].tolist() + ["Custom Manual Entry Mode"]
    selection = st.selectbox("Choose an entry from the production test set:", options)

    if selection and selection != "-- Select an Unseen Headline from Test Dataset --" and selection != "Custom Manual Entry Mode":
        matched_data = test_df[test_df['news_headline'] == selection].iloc[0]
        selected_headline = str(matched_data['news_headline'])
        # Handle concatenated features if present in test dataframe
        if 'news_source_text' in test_df.columns and pd.notna(matched_data['news_source_text']):
            selected_headline += " " + str(matched_data['news_source_text'])
        ground_truth_label = str(matched_data['label']).upper()

        # Ground Truth Information Panel Display Requirement
        st.markdown(f"""
            <div class='metadata-panel' style='border-left: 5px solid #4338ca;'>
                💡 <b>Dataset Ground Truth Label:</b> <span style='color: #4338ca; font-weight:bold;'>{ground_truth_label}</span>
                <br><small style='color:#64748b;'>This is the pristine evaluation label originally verified by fact-checkers.</small>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning(
        "⚠️ 'production_test_set.csv' not found. Defaulting to custom manual input mode only.")

# Dynamic input text box responding to selections
claim_text = st.text_area(
    "Claim Content Body / Text Area Input",
    value=selected_headline,
    height=150,
    placeholder="Select a headline above or type here to start analysis..."
)

st.markdown("---")
st.subheader("Step 2: Execution & Pipeline Metrics")

if st.button("Run System Verification Pipeline", type="primary"):
    if not claim_text.strip():
        st.warning("Input input area cannot be completely empty.")
    else:
        with st.spinner("Processing text features and tallying multi-classifier votes..."):
            # 1. Linguistic transformation layers
            cleaned_text = production_text_cleaner(claim_text)
            vectorized_input = vectorizer.transform([cleaned_text])

            # 2. Extract votes from internal estimators
            votes = []
            for name, clf in ensemble_model.named_estimators_.items():
                votes.append(clf.predict(vectorized_input)[0])

            # 3. Resolve system majority prediction
            majority_prediction_numeric = ensemble_model.predict(vectorized_input)[0]
            predicted_verdict = str(
                label_encoder.inverse_transform([majority_prediction_numeric])[0]).upper()

            # 4. Consensus Metrics
            agreeing_models = votes.count(majority_prediction_numeric)
            model_confidence = (agreeing_models / len(votes)) * 100

            # Output Presentation Rendering Split
            col_verdict, col_metrics = st.columns(2)

            with col_verdict:
                st.markdown("**Calculated Model Prediction:**")
                if predicted_verdict == "FALSE":
                    st.markdown(f"<div class='verdict-box verdict-false'>{predicted_verdict}</div>",
                                unsafe_allow_html=True)
                elif predicted_verdict == "MISLEADING":
                    st.markdown(
                        f"<div class='verdict-box verdict-misleading'>{predicted_verdict}</div>",
                        unsafe_url_allowed=True)
                else:
                    st.markdown(f"<div class='verdict-box verdict-true'>{predicted_verdict}</div>",
                                unsafe_url_allowed=True)

            with col_metrics:
                st.markdown("**Ensemble Engine Diagnostics:**")
                st.metric("Model Voting Consensus", f"{model_confidence:.1f}%")
                st.caption(
                    f"Agreement Ratio: {agreeing_models} out of 4 baseline model estimators flagged this label.")

            # --- DYNAMIC SYSTEM ALIGNMENT CHART REQUIREMENT ---
            if ground_truth_label:
                st.markdown("#### Ground-Truth vs. Model Prediction Comparison")

                # Check for label parity matches
                system_is_correct = (predicted_verdict == ground_truth_label)
                accuracy_comparison_pct = 100.0 if system_is_correct else 0.0

                # Format a horizontal bar chart directly mapping similarities
                comparison_chart_data = pd.DataFrame({
                    "System Metrics": ["Dataset Target", "Model Confidence"],
                    "Percentage (%)": [100.0, model_confidence]
                })

                # Use Streamlit native horizontal chart formatting
                st.bar_chart(
                    data=comparison_chart_data,
                    x="System Metrics",
                    y="Percentage (%)",
                    color="#4338ca" if system_is_correct else "#ef4444",
                    use_container_width=True
                )

                # Output analytical metrics interpretation
                if system_is_correct:
                    st.success(
                        f"🎯 **Validation Match:** System architecture cleanly matched the validation test row label with a validation accuracy index of **100%**.")
                else:
                    st.error(
                        f"❌ **System Divergence Detected:** The ensemble network predicted **{predicted_verdict}** against the target dataset label (**{ground_truth_label}**). The current model feature configuration mismatch yields an error offset profile.")

            # Natural Language Processing Pipeline Log Traces
            with st.expander("Linguistic Feature Inspection Logs"):
                st.text_style = "font-family: monospace; font-size: 0.85rem;"
                st.markdown(f"**Cleaned Tokens Input to Vectorizer Space:**")
                st.code(
                    cleaned_text if cleaned_text else "[Linguistic features reduced to null matrix after processing stop-words]")

st.markdown(
    "<div class='footer-text'>Thesis Methodology Operational Layout Prototype Core — Streamlit App Framework Engine.</div>",
    unsafe_allow_html=True)