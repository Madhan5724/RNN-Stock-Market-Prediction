# Real-Time NSE/BSE Stock Market Prediction Using RNN

## Project Overview

This project uses a Recurrent Neural Network (RNN) to analyse sequential stock market data and predict the next stock price.

The project is based on the Kaggle dataset:

**1M+ Real Time Stock Market Data [NSE/BSE]**

## Activity Requirement

**Variant B:** Use the SECOND HALF of the rows/date range in the dataset.

This requirement is implemented in `rnn_model.py`.

## Project Structure

```text
RNN_Stock_Market_Project/
│
├── dataset/
│   ├── instruments.csv
│   └── log_info.csv
│
├── models/
├── outputs/
├── rnn_model.py
├── check_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Copy the following Kaggle files into the `dataset` folder:

- `instruments.csv`
- `log_info.csv`

## Check Dataset

```bash
python check_data.py
```

## Run the RNN Model

```bash
python rnn_model.py
```

## Pipeline

1. Load stock market dataset
2. Apply Variant B
3. Use the second half of the dataset
4. Detect/select a stock price column
5. Normalize values using MinMaxScaler
6. Create 60-step sequences
7. Split data into 80% training and 20% testing
8. Train a Simple RNN
9. Predict stock prices
10. Evaluate using MSE and MAE
11. Generate Actual vs Predicted graph
12. Save the trained model

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## RNN Architecture

```text
60 Previous Price Values
        ↓
SimpleRNN (50 neurons)
        ↓
Dense (25 neurons)
        ↓
Dense (1 neuron)
        ↓
Predicted Next Price
```

## GitHub Upload

```bash
git init
git add .
git commit -m "Initial RNN stock market prediction project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The dataset and generated trained model are ignored by `.gitignore` to keep the repository lightweight.
