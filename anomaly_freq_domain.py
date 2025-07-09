import os
import logging
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

# --- Configuration ---
DATA_DIR = Path('data/bearing_data')
OUTPUT_CSV = Path('Averaged_BearingTest_Dataset.csv')
SCALER_FILENAME = Path("scaler_data_fft.gz")
MODEL_FILENAME = Path("Cloud_model_fft.h5")
EPOCHS = 100
BATCH_SIZE = 10

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_or_preprocess_data(data_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Load preprocessed data or process raw data files."""
    if output_csv.exists():
        logging.info(f"Found preprocessed data at '{output_csv}'. Loading from file.")
        return pd.read_csv(output_csv, index_col='timestamp', parse_dates=True)
    logging.info(f"No preprocessed data found. Processing raw data from '{data_dir}'...")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    records = []
    for filename in sorted(data_dir.iterdir()):
        if len(filename.name.split('.')) == 6:
            try:
                pd.to_datetime(filename.name, format='%Y.%m.%d.%H.%M.%S')
                dataset = pd.read_csv(filename, sep='\t', header=None)
                dataset_mean_abs = np.array(dataset.abs().mean())
                records.append((filename.name, *dataset_mean_abs))
            except Exception:
                continue
    if not records:
        raise ValueError("No valid data files found.")
    merged_data = pd.DataFrame.from_records(
        records,
        columns=['timestamp', 'Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
    )
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.set_index('timestamp').sort_index()
    merged_data.to_csv(output_csv)
    logging.info(f"Processed data saved to '{output_csv}'.")
    return merged_data

def plot_bearings(data: pd.DataFrame, title: str):
    plt.figure(figsize=(14, 6), dpi=80)
    for i, color in zip(range(1, 5), ['blue', 'red', 'green', 'black']):
        plt.plot(data[f'Bearing {i}'], label=f'Bearing {i}', color=color, linewidth=1)
    plt.legend(loc='lower left')
    plt.title(title, fontsize=16)
    plt.show()

def apply_fft(data: pd.DataFrame):
    fft = np.fft.rfft(data.values, axis=0)
    return fft, np.abs(fft)

def plot_fft(fft, title: str):
    plt.figure(figsize=(14, 6), dpi=80)
    for i, color in zip(range(4), ['blue', 'red', 'green', 'black']):
        plt.plot(fft[:, i].real, label=f'Bearing {i+1}', color=color, linewidth=1)
    plt.legend(loc='lower left')
    plt.title(title, fontsize=16)
    plt.show()

def scale_data(train, test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train)
    X_test_scaled = scaler.transform(test)
    joblib.dump(scaler, SCALER_FILENAME)
    logging.info(f"Scaler for FFT features saved to {SCALER_FILENAME}")
    return X_train_scaled, X_test_scaled

def reshape_for_lstm(X):
    return X.reshape(X.shape[0], 1, X.shape[1])

def autoencoder_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    x = LSTM(4, activation='relu', return_sequences=False)(x)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(4, activation='relu', return_sequences=True)(x)
    x = LSTM(16, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    return Model(inputs, outputs)

def plot_loss_distribution(loss, title):
    plt.figure(figsize=(12, 6))
    sns.histplot(loss, bins=50, kde=True, color='blue')
    plt.title(title)
    plt.show()

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(10)
    np.random.seed(10)

    # Load and preprocess data
    try:
        merged_data = load_or_preprocess_data(DATA_DIR, OUTPUT_CSV)
    except Exception as e:
        logging.error(e)
        return

    logging.info(f"Dataset shape: {merged_data.shape}")
    logging.info(f"\n{merged_data.head()}")

    # Train-test split
    train = merged_data['2004-02-12 10:52:39':'2004-02-15 12:52:39']
    test = merged_data['2004-02-15 12:52:39':]
    logging.info(f"Training dataset shape: {train.shape}")
    logging.info(f"Test dataset shape: {test.shape}")

    plot_bearings(train, 'Bearing Sensor Training Data')

    # FFT
    logging.info("--- Applying Fast Fourier Transform (FFT) ---")
    train_fft, X_train_fft = apply_fft(train)
    test_fft, X_test_fft = apply_fft(test)
    logging.info(f"Shape of FFT features for training: {X_train_fft.shape}")

    plot_fft(train_fft, 'Bearing Sensor Training Frequency Data')
    plot_fft(test_fft, 'Bearing Sensor Test Frequency Data')

    # Scaling
    X_train_scaled, X_test_scaled = scale_data(X_train_fft, X_test_fft)
    X_train_reshaped = reshape_for_lstm(X_train_scaled)
    X_test_reshaped = reshape_for_lstm(X_test_scaled)
    logging.info(f"Training sequences shape: {X_train_reshaped.shape}")
    logging.info(f"Test sequences shape: {X_test_reshaped.shape}")

    # Model
    model = autoencoder_model(X_train_reshaped.shape[1:])
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    logging.info("--- Training Model on FFT Features ---")
    history = model.fit(
        X_train_reshaped, X_train_reshaped,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.05, verbose=1
    )

    # Threshold
    logging.info("--- Calculating Anomaly Threshold ---")
    X_pred_train = model.predict(X_train_reshaped)
    train_mae_loss = np.mean(np.abs(X_pred_train - X_train_reshaped), axis=(1, 2))
    plot_loss_distribution(train_mae_loss, 'Loss Distribution on Training Data (FFT Features)')
    threshold = np.percentile(train_mae_loss, 99)
    logging.info(f"Anomaly threshold (99th percentile of training loss): {threshold:.4f}")

    # Test evaluation
    logging.info("--- Evaluating on Test Data ---")
    X_pred_test = model.predict(X_test_reshaped)
    test_mae_loss = np.mean(np.abs(X_pred_test - X_test_reshaped), axis=(1, 2))
    test_score_df = pd.DataFrame(index=test.index)
    test_score_df['Loss_mae'] = test_mae_loss
    test_score_df['Threshold'] = threshold
    test_score_df['Anomaly'] = test_score_df['Loss_mae'] > test_score_df['Threshold']
    anomalies = test_score_df[test_score_df['Anomaly']]
    logging.info(f"\nDetected anomalies:\n{anomalies}")

    # Plot anomalies
    plt.figure(figsize=(16, 8))
    test_score_df[['Loss_mae', 'Threshold']].plot(logy=True, ylim=[1e-2, 1e2])
    plt.title('Anomaly Detection on Test Data (FFT Features)')
    plt.show()

    # Save model
    model.save(MODEL_FILENAME)
    logging.info(f"Model trained on FFT features saved to {MODEL_FILENAME}")

if __name__ == "__main__":
    main()
