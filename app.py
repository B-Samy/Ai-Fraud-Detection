
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


model = joblib.load("model.pkl")


st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.caption("Made by Shaheer Rangrej 🤝 Ismile Mirza")

st.title("💳 AI Fraud Transaction Detection")
st.write("Analyze a transaction and estimate the probability of fraudulent activity.")

st.sidebar.header("Transaction Details")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0,
    value=50000,
    step=5000
)

oldbalanceOrg = st.sidebar.number_input(
    "Sender Balance Before",
    min_value=0,
    value=60000,
    step=5000
)

newbalanceOrig = st.sidebar.number_input(
    "Sender Balance After",
    min_value=0,
    value=10000,
    step=5000
)

oldbalanceDest = st.sidebar.number_input(
    "Receiver Balance Before",
    min_value=0,
    value=20000,
    step=5000
)

newbalanceDest = st.sidebar.number_input(
    "Receiver Balance After",
    min_value=0,
    value=70000,
    step=5000
)

predict = st.sidebar.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)

if not predict:
    st.info("Enter the transaction details from the sidebar and click **Analyze Transaction**.")

    st.subheader("How the Detection Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Transaction Types", "5")
        st.caption("CASH_IN, CASH_OUT, DEBIT, PAYMENT and TRANSFER")

    with col2:
        st.metric("AI Features", "9")
        st.caption("Transaction and balance-related features")

    with col3:
        st.metric("Output", "0 / 1")
        st.caption("0 = Legitimate, 1 = Fraud")

else:
    original_sender_balance = oldbalanceOrg - newbalanceOrig

    Dest_reciever_balance = newbalanceDest - oldbalanceDest

    amount_old_bal = (amount / (oldbalanceOrg + 1)) * 100 

    new_transaction = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "original_sender_balance": original_sender_balance,
        "Dest_reciever_balance": Dest_reciever_balance,
        "ammount/old_bal": amount_old_bal
    }

    transaction_df = pd.DataFrame([new_transaction])

    transaction_df = pd.get_dummies(
        transaction_df,
        columns=["type"]
    )

    if hasattr(model, "feature_names_in_"):
        transaction_df = transaction_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )
    else:
        st.error("The saved model does not contain feature names. The training feature order must be provided.")
        st.stop()

    prediction = model.predict(transaction_df)[0]
    probability = model.predict_proba(transaction_df)[0]

    legitimate_probability = probability[0] * 100
    fraud_probability = probability[1] * 100

    st.subheader("Detection Result")

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION")
        st.write("The model classified this transaction as potentially fraudulent.")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")
        st.write("The model classified this transaction as legitimate.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

    with col3:
        st.metric(
            "Transaction Amount",
            f"₹{amount:,.0f}"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Model Confidence")

        probability_df = pd.DataFrame(
            {
                "Class": ["Legitimate", "Fraud"],
                "Probability": [
                    legitimate_probability,
                    fraud_probability
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )

    with col2:
        st.subheader("💰 Balance Movement")

        balance_df = pd.DataFrame(
            {
                "Balance": [
                    oldbalanceOrg,
                    newbalanceOrig,
                    oldbalanceDest,
                    newbalanceDest
                ]
            },
            index=[
                "Sender Before",
                "Sender After",
                "Receiver Before",
                "Receiver After"
            ]
        )

        st.bar_chart(balance_df)

    st.divider()

    st.subheader("🔎 Engineered Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sender Balance Change",
            f"₹{original_sender_balance:,.0f}"
        )

    with col2:
        st.metric(
            "Receiver Balance Change",
            f"₹{Dest_reciever_balance:,.0f}"
        )

    with col3:
        st.metric(
            "Amount / Old Balance",
            f"{amount_old_bal:.2f}%"
        )

    st.subheader("📋 Transaction Analysis")

    analysis_df = pd.DataFrame(
        {
            "Metric": [
                "Transaction Type",
                "Amount",
                "Sender Balance Before",
                "Sender Balance After",
                "Receiver Balance Before",
                "Receiver Balance After",
                "Sender Balance Change",
                "Receiver Balance Change",
                "Amount / Old Balance"
            ],
            "Value": [
                transaction_type,
                f"₹{amount:,.0f}",
                f"₹{oldbalanceOrg:,.0f}",
                f"₹{newbalanceOrig:,.0f}",
                f"₹{oldbalanceDest:,.0f}",
                f"₹{newbalanceDest:,.0f}",
                f"₹{original_sender_balance:,.0f}",
                f"₹{Dest_reciever_balance:,.0f}",
                f"{amount_old_bal:.2f}%"
            ]
        }
    )

    st.dataframe(
        analysis_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Transaction Risk")

    if fraud_probability >= 75:
        st.error("🔴 High Fraud Risk")
        st.write("The model assigns a high probability to the fraud class.")

    elif fraud_probability >= 40:
        st.warning("🟠 Moderate Fraud Risk")
        st.write("The model sees some suspicious characteristics, but the prediction is not strongly classified as fraud.")

    else:
        st.success("🟢 Low Fraud Risk")
        st.write("The model assigns a higher probability to the legitimate class.")

    st.progress(
        int(min(max(fraud_probability, 0), 100))
    )

    st.caption(
        f"Fraud probability: {fraud_probability:.2f}%"
    )

    with st.expander("View Model Input"):
        st.dataframe(
            transaction_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("About This Prediction"):
        st.write(
            "The model does not determine fraud from one rule such as transaction amount. "
            "It uses the feature patterns learned during training. "
            "The probability shown above represents the model's estimated probability for each class."
        )
