
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go 

# Load model and dataset
model = joblib.load('churn_model.pkl')
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Title
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Analyze churn data, predict customer churn, and view insights.")

# Sidebar for navigation
menu = ["Prediction", "EDA Insights"]
choice = st.sidebar.selectbox("Select Section", menu)

# =================== PREDICTION SECTION ===================
if choice == "Prediction":
    st.header("🔍 Predict Churn for a Customer")

    # Collect user input
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior = st.selectbox("Senior Citizen", [0, 1], key="senior")
    tenure = st.slider("Tenure (Months)", 0, 72, 12, key="tenure")
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, key="monthly")
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0, key="total")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], key="payment")
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")

    # Convert to dataframe
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment,
        'PaperlessBilling': paperless
    }
    input_df = pd.DataFrame([input_data])
    st.write("### Input Summary:")
    st.write(input_df)

    # Preprocess input same as training
    input_df = pd.get_dummies(input_df)
    model_cols = model.get_booster().feature_names  # XGBoost feature names
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_cols]

    # Predict churn
    if st.button("Predict Churn", key="predict_btn"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.subheader("Prediction Result:")
        if pred == 1:
            st.error(f"⚠️ Customer is likely to churn. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Customer is likely to stay. (Probability: {prob:.2f})")

        # Visualization - Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prob > 0.5 else "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 100], 'color': "pink"}
                   ]}
        ))
        st.plotly_chart(fig)

        # Bar chart for probability comparison
        st.write("### Probability Comparison")
        fig_bar = go.Figure(data=[
            go.Bar(name='Stay', x=['Customer'], y=[(1 - prob) * 100], marker_color='green'),
            go.Bar(name='Churn', x=['Customer'], y=[prob * 100], marker_color='red')
        ])
        fig_bar.update_layout(barmode='group', yaxis_title="Probability (%)")
        st.plotly_chart(fig_bar)

# =================== EDA SECTION ===================
elif choice == "EDA Insights":
    st.header("📈 Churn Data Insights")

    # Churn distribution
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=data, palette='coolwarm', ax=ax1)
    st.pyplot(fig1)

    # Churn by Contract
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=data, palette='viridis', ax=ax2)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    # Churn by Internet Service
    fig3, ax3 = plt.subplots()
    sns.countplot(x='InternetService', hue='Churn', data=data, palette='magma', ax=ax3)
    st.pyplot(fig3)

    st.write("More insights can be added: tenure analysis, payment method, etc.")
