import streamlit as st
import joblib
import numpy as np


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #1f2937;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.positive {
    background: #dcfce7;
    color: #166534;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.negative {
    background: #fee2e2;
    color: #991b1b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.neutral {
    background: #fef3c7;
    color: #92400e;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 AI Sentiment Analyzer")

    st.markdown("---")

    st.markdown("""
### 📌 About

This application uses:

- TF-IDF Vectorization
- LinearSVC
- Natural Language Processing
- Machine Learning

### 🎯 Sentiment Classes

😊 Positive

😞 Negative

😐 Neutral
""")

    st.markdown("---")

    st.info(
        "Enter a review or sentence to analyze its sentiment."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">💬 AI Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze text sentiment using Machine Learning & NLP'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT
# =========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### ✍️ Enter Your Text")

text = st.text_area(
    "Text",
    height=160,
    placeholder="Example: The service was average.",
    label_visibility="collapsed"
)


# =========================================================
# EXAMPLES
# =========================================================

st.markdown("### 💡 Try an Example")

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "😊 Positive",
        use_container_width=True
    ):

        text = (
            "I absolutely love this product! "
            "It is amazing and I am very happy with it."
        )


with col2:

    if st.button(
        "😞 Negative",
        use_container_width=True
    ):

        text = (
            "I am extremely disappointed with this product. "
            "The quality is terrible."
        )


with col3:

    if st.button(
        "😐 Neutral",
        use_container_width=True
    ):

        # This example was tested against your actual model
        text = "The service was average."


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning("⚠️ Please enter some text.")

    else:

        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

        text_vector = vectorizer.transform([text])


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        original_prediction = model.predict(text_vector)[0]


        # -------------------------------------------------
        # DECISION SCORES
        # -------------------------------------------------

        scores = model.decision_function(text_vector)[0]

        classes = model.classes_


        # Create dictionary:
        # Negative -> score
        # Neutral -> score
        # Positive -> score

        score_dict = {
            str(classes[i]): float(scores[i])
            for i in range(len(classes))
        }


        # -------------------------------------------------
        # FIND HIGHEST SCORE
        # -------------------------------------------------

        highest_class = str(
            classes[np.argmax(scores)]
        )

        highest_score = float(
            np.max(scores)
        )


        # -------------------------------------------------
        # FIND SECOND HIGHEST SCORE
        # -------------------------------------------------

        sorted_scores = np.sort(scores)[::-1]

        first_score = sorted_scores[0]
        second_score = sorted_scores[1]

        score_difference = first_score - second_score


        # -------------------------------------------------
        # IMPROVED NEUTRAL LOGIC
        # -------------------------------------------------

        prediction = highest_class


        # If Neutral is the strongest class,
        # use Neutral directly.

        if highest_class == "Neutral":

            prediction = "Neutral"


        else:

            # If prediction is very close between classes,
            # consider it Neutral.

            if score_difference < 0.25:

                prediction = "Neutral"


            # If the model's strongest score is weak
            # and the difference is small, use Neutral.

            elif highest_score < 0.15 and score_difference < 0.60:

                prediction = "Neutral"


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        # Softmax-like score for displaying relative
        # model confidence. This is NOT a probability.

        exp_scores = np.exp(
            scores - np.max(scores)
        )

        confidence_scores = (
            exp_scores / np.sum(exp_scores)
        )

        confidence = float(
            np.max(confidence_scores)
        )

        confidence_percentage = confidence * 100


        # If we changed an ambiguous prediction to Neutral,
        # don't show misleadingly high confidence.

        if prediction == "Neutral" and highest_class != "Neutral":

            confidence_percentage = min(
                confidence_percentage,
                65
            )


        # =================================================
        # DISPLAY SENTIMENT
        # =================================================

        if prediction == "Positive":

            emoji = "😊"
            css_class = "positive"

        elif prediction == "Negative":

            emoji = "😞"
            css_class = "negative"

        else:

            emoji = "😐"
            css_class = "neutral"


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.markdown("## 📊 Analysis Result")

        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # SENTIMENT
        # -------------------------------------------------

        with col1:

            st.markdown(
                '<div class="card">'
                '<h3>Detected Sentiment</h3>'
                f'<div class="{css_class}">'
                f'{emoji} {prediction.upper()}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        with col2:

            st.markdown(
                '<div class="card">'
                '<h3>🎯 Model Confidence</h3>',
                unsafe_allow_html=True
            )

            st.metric(
                "Confidence Score",
                f"{confidence_percentage:.2f}%"
            )

            st.progress(
                min(
                    max(
                        confidence_percentage / 100,
                        0.0
                    ),
                    1.0
                )
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.markdown("### 🧠 Interpretation")

        if prediction == "Positive":

            st.success(
                "The model predicts a **positive sentiment**."
            )

        elif prediction == "Negative":

            st.error(
                "The model predicts a **negative sentiment**."
            )

        else:

            st.warning(
                "The text appears to express a "
                "**neutral or balanced sentiment**."
            )


        # =================================================
        # CLASS SCORES
        # =================================================

        st.markdown("### 📈 Model Decision Scores")

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:

            st.metric(
                "😞 Negative",
                f"{score_dict.get('Negative', 0):.3f}"
            )

        with score_col2:

            st.metric(
                "😐 Neutral",
                f"{score_dict.get('Neutral', 0):.3f}"
            )

        with score_col3:

            st.metric(
                "😊 Positive",
                f"{score_dict.get('Positive', 0):.3f}"
            )


        # =================================================
        # INPUT TEXT
        # =================================================

        st.markdown("### 📝 Analyzed Text")

        st.markdown(
            f"""
            <div style="
                background:white;
                padding:20px;
                border-radius:15px;
                border-left:5px solid #6366f1;
            ">
            {text}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        margin-top:50px;
        padding:20px;
    ">
        🤖 AI Sentiment Analyzer
        <br>
        Python • TF-IDF • LinearSVC • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)