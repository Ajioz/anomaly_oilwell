import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import io
from contextlib import redirect_stdout

# --- Configuration ---
DATA_DIR = 'data/bearing_data'
OUTPUT_CSV = 'Averaged_BearingTest_Dataset.csv'
SCALER_FILENAME = "scaler_data.gz"
MODEL_FILENAME = "Cloud_model.h5"

# Set random seed for reproducibility
tf.random.set_seed(10)
np.random.seed(10)

# --- Model & Training Hyperparameters ---
TIME_STEPS = 10  # Number of time steps in each sequence
EPOCHS = 100
BATCH_SIZE = 16


def create_sequences(X: np.ndarray, time_steps: int = 1):
    """
    Creates time-series sequences from a 2D array.
    """
    Xs = []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)


def generate_report(model, hyperparameters, report_path="reports/training_report.txt"):
    """
    Generates a text report with a model summary and a list of hyperparameters.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        hyperparameters (dict): A dictionary of hyperparameters for logging.
        report_path (str or Path): The path to save the report file.
    """
    report_path = Path(report_path)
    # Ensure the parent directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture model.summary() output into a string
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    model_summary_str = stream.getvalue()

    with open(report_path, 'w', encoding='utf-8') as f:
        # --- Model Summary Section ---
        f.write("="*80 + "\n")
        f.write("Model Summary Table\n")
        f.write("="*80 + "\n")
        f.write(model_summary_str)
        f.write("\n\n")

        # --- Hyperparameters Section ---
        f.write("="*80 + "\n")
        f.write("Full Hyperparameter List\n")
        f.write("="*80 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"{key:<25}: {value}\n")
        f.write("\n")

    print(f"Training report saved to: {report_path}")


# load, average and merge sensor samples
print("Loading and preprocessing data...")
records = []

# Check if the data directory exists
if not os.path.isdir(DATA_DIR):
    print(f"Error: The directory '{DATA_DIR}' was not found. Please ensure it has been unzipped correctly.")
    exit()

file_list = sorted(os.listdir(DATA_DIR))

if not file_list:
    print(f"Error: The directory '{DATA_DIR}' is empty. No files found to process.")
    exit()

for filename in file_list:
    # Skip non-data files by checking if the filename can be parsed as a datetime.
    # This avoids issues with hidden files like '.DS_Store' or other non-data files.
    try:
        pd.to_datetime(filename, format='%Y.%m.%d.%H.%M.%S')
        dataset_path = os.path.join(DATA_DIR, filename)
        # Assuming no header in the raw files
        dataset = pd.read_csv(dataset_path, sep='\t', header=None)
        dataset_mean_abs = np.array(dataset.abs().mean())
        records.append((filename, *dataset_mean_abs))
    except (ValueError, TypeError):
        # Skip files that don't match the expected datetime format
        continue

# Create DataFrame efficiently
merged_data = pd.DataFrame.from_records(records,
    columns=['timestamp', 'Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
)
# transform data file index to datetime and sort in chronological order
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.set_index('timestamp').sort_index()

# --- Sanity Check for Loaded Data ---
if merged_data.empty:
    print("\n[ERROR] No data was loaded. The 'merged_data' DataFrame is empty.")
    print(f"Please check the '{DATA_DIR}' directory and ensure it contains valid data files.")
    exit()
else:
    # Save the merged data only if it's not empty
    merged_data.to_csv(OUTPUT_CSV)
    print("\nDataset shape:", merged_data.shape)
    print(merged_data.head())
    print(f"\nData successfully loaded for the date range: {merged_data.index.min()} to {merged_data.index.max()}")

# Split data into training and testing sets
train = merged_data['2004-02-12 10:52:39': '2004-02-15 12:52:39']
test = merged_data['2004-02-15 12:52:39':]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

if train.empty or test.empty:
    print("\n[ERROR] The training or test dataset is empty after slicing by date.")
    print("Please ensure the hardcoded date ranges overlap with the loaded data range shown above.")
    exit()

# --- Plotting Initial Data ---
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['Bearing 1'], label='Bearing 1', color='blue', linewidth=1)
ax.plot(train['Bearing 2'], label='Bearing 2', color='red', linewidth=1)
ax.plot(train['Bearing 3'], label='Bearing 3', color='green', linewidth=1)
ax.plot(train['Bearing 4'], label='Bearing 4', color='black', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)
plt.show()

# --- Data Scaling ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
joblib.dump(scaler, SCALER_FILENAME)
print(f"Scaler saved to {SCALER_FILENAME}")


# --- Create Time-Series Sequences ---
X_train_seq = create_sequences(X_train, TIME_STEPS)
X_test_seq = create_sequences(X_test, TIME_STEPS)
print("Training sequences shape:", X_train_seq.shape)
print("Test sequences shape:", X_test_seq.shape)


# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model


# create the autoencoder model
model = autoencoder_model(X_train_seq)
model.compile(optimizer='adam', loss='mae')
model.summary()

# --- Generate Report ---
print("\n--- Generating Training Report ---")
# Collate hyperparameters for the report
hyperparameters = {
    'TIME_STEPS': TIME_STEPS,
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'VALIDATION_SPLIT': 0.05,
    'LSTM_UNITS_L1': 16,
    'LSTM_UNITS_L2': 4,
    'REGULARIZATION': 'l2(0.00)',
    'OPTIMIZER': 'adam',
    'LOSS_FUNCTION': 'mae'
}
generate_report(model, hyperparameters)

# fit the model to the data
print("\n--- Training Model ---")
history = model.fit(X_train_seq, X_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.05, verbose=1).history


# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

# --- Determine Anomaly Threshold ---
print("\n--- Calculating Anomaly Threshold ---")
X_pred_train = model.predict(X_train_seq)
# Calculate the Mean Absolute Error for each sequence.
# The result is a 1D array of loss values, one for each input sequence.
train_mae_loss = np.mean(np.abs(X_pred_train - X_train_seq), axis=(1, 2))

# Set threshold to the 99th percentile of the training loss distribution
threshold = np.percentile(train_mae_loss, 99)
print(f"Anomaly threshold (99th percentile of training loss): {threshold:.4f}")

# Plot the loss distribution of the training set to visualize the threshold
fig, ax = plt.subplots(figsize=(12, 6), dpi=80)
sns.histplot(train_mae_loss, bins=50, kde=True, color='blue', ax=ax)
ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
ax.set_title('Loss Distribution on Training Data', fontsize=16)
ax.set_xlabel('Reconstruction Error (MAE)')
ax.set_ylabel('Frequency')
ax.legend()
plt.show()

# --- Evaluate on Test Data ---
print("\n--- Evaluating on Test Data ---")
X_pred_test = model.predict(X_test_seq)
test_mae_loss = np.mean(np.abs(X_pred_test - X_test_seq), axis=(1, 2))

# Create a DataFrame to store the loss and anomaly info.
# The index must be shifted by TIME_STEPS to align with the sequences.
train_score_df = pd.DataFrame(index=train.index[TIME_STEPS:])
train_score_df['Loss_mae'] = train_mae_loss
train_score_df['Threshold'] = threshold
train_score_df['Anomaly'] = train_score_df['Loss_mae'] > train_score_df['Threshold']

test_score_df = pd.DataFrame(index=test.index[TIME_STEPS:])
test_score_df['Loss_mae'] = test_mae_loss
test_score_df['Threshold'] = threshold
test_score_df['Anomaly'] = test_score_df['Loss_mae'] > test_score_df['Threshold']

# Combine train and test scores for plotting
scored = pd.concat([train_score_df, test_score_df])

# --- Plot Anomalies ---
# Filter for anomalies only in the test set for reporting
anomalies = test_score_df[test_score_df['Anomaly']]
print("\nDetected anomalies:")
print(anomalies)

# Plot the loss and anomalies
fig, ax = plt.subplots(figsize=(16, 9), dpi=80)
scored['Loss_mae'].plot(ax=ax, label='Reconstruction Loss', color='blue')
scored['Threshold'].plot(ax=ax, label='Threshold', color='red', linestyle='--')
ax.scatter(anomalies.index, anomalies['Loss_mae'], color='red', marker='o', s=50, label='Anomaly')
ax.set_title('Anomaly Detection: Reconstruction Loss Over Time', fontsize=16)
ax.set_ylabel('Mean Absolute Error (log scale)')
ax.set_yscale('log')
ax.set_ylim(1e-2, 1e2)
ax.legend(loc='upper left')
plt.show()

# save all model information, including weights, in h5 format
model.save(MODEL_FILENAME)
print(f"Model saved to {MODEL_FILENAME}")
