import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained models and transformer
lr_model = joblib.load("logistic_regression_model.pkl")
rfc_model = joblib.load("random_forest_model.pkl")
transformer = joblib.load("transformer.pkl")  # Load transformer

def main():
    st.title("Loan Approval Prediction App")

    # User inputs
    person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Person Income", min_value=1000, value=50000)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, value=5)
    loan_amnt = st.number_input("Loan Amount", min_value=500, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=5.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.2)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)

    # Categorical Inputs
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT"])

    # **Fix: Include missing categorical columns**
    person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["No", "Yes"])
    person_gender = st.selectbox("Gender", ["Female", "Male"])

    # **Apply the same encoding as during training**
    education_mapping = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    person_education = education_mapping[person_education]

    gender_mapping = {"Female": 0, "Male": 1}
    person_gender = gender_mapping[person_gender]

    loan_default_mapping = {"No": 0, "Yes": 1}
    previous_loan_defaults_on_file = loan_default_mapping[previous_loan_defaults_on_file]

    # Convert inputs into a DataFrame (with all expected columns)
    input_data = pd.DataFrame([[person_age, person_income, person_emp_exp, loan_amnt, 
                                loan_int_rate, loan_percent_income, cb_person_cred_hist_length, 
                                credit_score, person_home_ownership, loan_intent,
                                person_education, previous_loan_defaults_on_file, person_gender]],
                              columns=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
                                       'credit_score', 'person_home_ownership', 'loan_intent',
                                       'person_education', 'previous_loan_defaults_on_file', 'person_gender'])

    # **Apply the same transformations as during training**
    input_data_transformed = transformer.transform(input_data)

    # Predictions
    if st.button("Predict Loan Approval"):
        lr_prediction = lr_model.predict(input_data_transformed)[0]
        rfc_prediction = rfc_model.predict(input_data_transformed)[0]

        st.subheader("Prediction Results")
        st.write(f"**Logistic Regression Prediction:** {'Approved' if lr_prediction == 1 else 'Rejected'}")
        st.write(f"**Random Forest Prediction:** {'Approved' if rfc_prediction == 1 else 'Rejected'}")

if __name__ == "__main__":
    main()
#python -m streamlit run FinScope.py9