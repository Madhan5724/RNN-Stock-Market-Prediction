import streamlit as st

st.set_page_config(
    page_title="RNN Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 RNN Stock Market Prediction")
st.subheader("Real-Time NSE/BSE Stock Market Prediction Using RNN")

st.write("""
This project uses a Recurrent Neural Network (RNN) to analyse
sequential historical stock market data and predict stock prices.
""")

st.divider()

st.header("About the Project")

st.write("""
The model uses:

• Historical stock market data

• Time-series preprocessing

• MinMax normalization

• 60-step sequences

• SimpleRNN deep learning model

• Actual vs Predicted stock price analysis
""")

st.header("Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🐍 Python")

with col2:
    st.info("🧠 TensorFlow / Keras")

with col3:
    st.info("📊 Pandas & NumPy")

st.divider()

st.success("RNN Stock Market Prediction Project Successfully Deployed 🚀")
