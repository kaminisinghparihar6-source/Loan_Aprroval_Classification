import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #6a11cb 100%);
    }
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #e0e7ff;
        margin-bottom: 25px;
        font-size: 1.05rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px 30px;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    .card h3 { color: #2a5298; margin-top: 0; }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff512f, #f09819);
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.7rem 0;
        border: none;
        border-radius: 12px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 18px rgba(240, 152, 25, 0.5);
        color: white;
    }
    .approved {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 22px; border-radius: 16px; text-align: center;
        color: white; font-size: 1.7rem; font-weight: 800;
    }
    .rejected {
        background: linear-gradient(90deg, #cb2d3e, #ef473a);
        padding: 22px; border-radius: 16px; text-align: center;
        color: white; font-size: 1.7rem; font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

# ---------------- Header ----------------
st.markdown('<div class="main-title">💰 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Apni details bharo aur turant pata karo loan approve hoga ya nahi 🚀</div>',
    unsafe_allow_html=True,
)

# ---------------- Input Form ----------------
st.markdown('<div class="card"><h3>👤 Personal Details</h3>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_emp_exp = st.number_input("Employment Experience (years)", 0, 60, 3)
with c2:
    person_education = st.selectbox(
        "Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card"><h3>🏦 Loan Details</h3>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    loan_amnt = st.number_input("Loan Amount", 500, 100000, 10000, step=500)
    loan_int_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 11.0, 0.1)
    loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2, 0.01)
with c4:
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, 5)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", ["No", "Yes"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Predict ----------------
if st.button("🔮 Predict Loan Status"):
    input_df = pd.DataFrame(
        [
            {
                "person_gender": person_gender,
                "person_education": person_education,
                "person_emp_exp": person_emp_exp,
                "person_home_ownership": person_home_ownership,
                "loan_amnt": loan_amnt,
                "loan_intent": loan_intent,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
            }
        ]
    )

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        # 1 = Approved, 0 = Rejected (dataset ke hisaab se)
        approved_idx = classes.index(1) if 1 in classes else len(classes) - 1
        approval_prob = proba[approved_idx] * 100

        st.markdown("---")
        if int(prediction) == 1:
            st.markdown('<div class="approved">✅ Loan APPROVED 🎉</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<div class="rejected">❌ Loan REJECTED</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<h4 style='color:white;text-align:center;'>Approval Probability: {approval_prob:.2f}%</h4>",
            unsafe_allow_html=True,
        )
        st.progress(float(approval_prob / 100))

        with st.expander("📋 Input Data dekho"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction me error: {e}")
        st.info("Check karo ki input columns ke naam training wale X.columns se match karte hain.")

st.markdown(
    "<p style='text-align:center;color:#c7d2fe;margin-top:30px;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
