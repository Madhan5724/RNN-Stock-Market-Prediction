import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "dataset/log_info.csv"

WINDOW_SIZE = 60
EPOCHS = 10
BATCH_SIZE = 32

# Set these manually only if automatic detection does not work.
PRICE_COLUMN = None
SYMBOL_COLUMN = None
SYMBOL_VALUE = None


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}\n"
        "Copy the Kaggle file log_info.csv into the dataset folder."
    )

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 60)
print("REAL-TIME NSE/BSE STOCK MARKET RNN PROJECT")
print("=" * 60)

print("\nOriginal Dataset Shape:", df.shape)
print("\nAvailable Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# OPTIONAL: SELECT ONE STOCK
# ============================================================

if SYMBOL_COLUMN is not None and SYMBOL_VALUE is not None:
    if SYMBOL_COLUMN not in df.columns:
        raise ValueError(
            f"Symbol column '{SYMBOL_COLUMN}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[df[SYMBOL_COLUMN] == SYMBOL_VALUE].copy()

    print("\nSelected Symbol:", SYMBOL_VALUE)
    print("Rows after symbol selection:", len(df))


# ============================================================
# VARIANT B
# USE SECOND HALF OF DATASET
# ============================================================

df = df.iloc[len(df) // 2:].copy().reset_index(drop=True)

print("\nVariant B Applied")
print("Using SECOND HALF of the dataset")
print("Rows after Variant B:", len(df))


# ============================================================
# AUTOMATIC PRICE COLUMN DETECTION
# ============================================================

if PRICE_COLUMN is None:

    preferred_names = [
        "close",
        "ltp",
        "last_price",
        "last",
        "price",
        "value",
        "current_price"
    ]

    lower_columns = {str(col).lower(): col for col in df.columns}

    detected_column = None

    for name in preferred_names:
        if name in lower_columns:
            detected_column = lower_columns[name]
            break

    if detected_column is None:

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_columns) == 0:
            raise ValueError(
                "No numeric column found automatically.\n"
                f"Available columns: {df.columns.tolist()}\n"
                "Set PRICE_COLUMN manually in the configuration section."
            )

        detected_column = numeric_columns[0]

    PRICE_COLUMN = detected_column


print("\nSelected Price Column:", PRICE_COLUMN)


# ============================================================
# PREPARE PRICE DATA
# ============================================================

if PRICE_COLUMN not in df.columns:
    raise ValueError(
        f"PRICE_COLUMN '{PRICE_COLUMN}' not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

data = pd.to_numeric(df[PRICE_COLUMN], errors="coerce").dropna().values.reshape(-1, 1)

print("Valid price records:", len(data))

if len(data) <= WINDOW_SIZE + 1:
    raise ValueError(
        "Not enough valid data to create sequences.\n"
        "Choose a different PRICE_COLUMN or reduce WINDOW_SIZE."
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# ============================================================
# CREATE TIME SERIES SEQUENCES
# ============================================================

X = []
y = []

for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

print("\nRNN Input Shape:", X.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# BUILD RNN MODEL
# ============================================================

model = Sequential([
    SimpleRNN(
        50,
        activation="relu",
        input_shape=(WINDOW_SIZE, 1)
    ),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\nModel Architecture:")
model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# ============================================================
# EVALUATION
# ============================================================

mse = mean_squared_error(actual, predictions)
mae = mean_absolute_error(actual, predictions)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("RNN Stock Price Prediction - Variant B")
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
plt.show()


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("RNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/training_loss.png", dpi=300)
plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

model.save("models/rnn_stock_model.keras")

print("\nModel saved successfully:")
print("models/rnn_stock_model.keras")

print("\nOutput graphs saved in outputs folder.")
