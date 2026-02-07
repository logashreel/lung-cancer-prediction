import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from datetime import date

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Lung Cancer Prediction System",
    layout="centered"
)

# ================= TITLE =================
st.title("🫁 Lung Cancer Prediction System")
st.write("Single-page dashboard for final survival prediction")

# ================= LOAD DATASET =================
df = pd.read_csv(
    r"D:\Project\Lung_cancer\dataset_med.csv",
    nrows=5000,
    low_memory=True
)

# ================= PREPROCESS DATA =================
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# ================= TRAIN MODEL =================
X = df.drop("survived", axis=1)
y = df["survived"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ================= USER INPUT =================
st.subheader("📝 Enter Patient Details")

patient_id = st.number_input("Patient ID", min_value=1, step=1)

age = st.number_input("Age", min_value=1, max_value=120, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])

country = st.text_input("Country")

diagnosis_date = st.date_input("Diagnosis Date", value=date.today())

cancer_stage = st.selectbox(
    "Cancer Stage",
    ["Stage I", "Stage II", "Stage III", "Stage IV"]
)

family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])

smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)

cholesterol_level = st.number_input(
    "Cholesterol Level (mg/dL)",
    min_value=100, max_value=400, step=1
)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
asthma = st.selectbox("Asthma", ["No", "Yes"])
cirrhosis = st.selectbox("Cirrhosis", ["No", "Yes"])
other_cancer = st.selectbox("Other Cancer History", ["No", "Yes"])

treatment_type = st.selectbox(
    "Treatment Type",
    ["Chemotherapy", "Radiation", "Surgery", "Combination"]
)

end_treatment_date = st.date_input(
    "End of Treatment Date", value=date.today()
)

# ================= CONVERT INPUT TO MODEL FORMAT =================
input_dict = {
    "id": patient_id,
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "country": country,
    "diagnosis_date": diagnosis_date.toordinal(),
    "cancer_stage": cancer_stage,
    "family_history": 1 if family_history == "Yes" else 0,
    "smoking_status": 1 if smoking_status == "Yes" else 0,
    "bmi": bmi,
    "cholesterol_level": cholesterol_level,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "asthma": 1 if asthma == "Yes" else 0,
    "cirrhosis": 1 if cirrhosis == "Yes" else 0,
    "other_cancer": 1 if other_cancer == "Yes" else 0,
    "treatment_type": treatment_type,
    "end_treatment_date": end_treatment_date.toordinal()
}

input_df = pd.DataFrame([input_dict])

# Encode categorical inputs
for col in input_df.columns:
    if input_df[col].dtype == "object":
        input_df[col] = le.fit_transform(input_df[col])

# ================= PREDICTION =================
if st.button("🔍 Predict Final Outcome"):

    prediction = model.predict(input_df)[0]

    st.subheader("🧾 Final Prediction Result")

    if prediction == 1:
        st.success("✅ Patient is likely to SURVIVE")
        risk = "LOW RISK"
        recommendation = (
            "Continue follow-ups, maintain healthy lifestyle, "
            "and attend regular medical checkups."
        )
    else:
        st.error("⚠️ Patient is at HIGH RISK")
        risk = "HIGH RISK"
        recommendation = (
            "Immediate specialist consultation required. "
            "Strict treatment monitoring advised."
        )

    # ================= FINAL SUMMARY =================
    st.subheader("📄 Final Patient Summary")
    st.write(f"""
    **Patient ID:** {patient_id}  
    **age:** {age}  
    **Gender:** {gender}  
    **Country:** {country}  
    **Cancer Stage:** {cancer_stage}  
    **Risk Level:** {risk}  
    **Prediction Status:** {'Survived' if prediction == 1 else 'Critical'}  
    **Recommendation:** {recommendation}
    """)

    # ================= REPORT DOWNLOAD =================
    report_df = pd.DataFrame({
        "Patient ID": [patient_id],
        "age": [age],
        "Gender": [gender],
        "Country": [country],
        "Cancer Stage": [cancer_stage],
        "Risk Level": [risk],
        "Prediction Result": ["Survived" if prediction == 1 else "High Risk"],
        "Recommendation": [recommendation]
    })

    st.download_button(
        label="⬇ Download Final Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="final_patient_report.csv",
        mime="text/csv"
    )

# ================= FOOTER =================
st.markdown("---")
st.caption("Lung Cancer Prediction System | Final Year Project")
