# Time-Series Anomaly Detection using LSTM Autoencoder

This project demonstrates how to use an LSTM (Long Short-Term Memory) Autoencoder to detect anomalies in time-series data. The primary example (`lstm_autoencoder.py`) uses historical stock price data for General Electric (GE) to identify periods of unusual market behavior.

A simpler, non-temporal example (`basic.py`) is also included, which uses a dense autoencoder on a synthetic dataset.

## How It Works

The core principle is to train a neural network to learn a compressed representation of "normal" time-series sequences.

1.  **Training on Normal Data**: An LSTM autoencoder is trained on a dataset containing only normal sequences (in this case, GE stock prices from before 2004). The model's objective is to reconstruct its own input as accurately as possible.
2.  **Learning Patterns**: The model learns the underlying patterns of normal data and, as a result, can reconstruct it with a very low error (Mean Absolute Error - MAE).
3.  **Statistical Thresholding**: A statistical threshold is calculated from the reconstruction errors on the normal training data (e.g., the 99th percentile). This threshold represents the maximum error expected for a "normal" sequence.
4.  **Detecting Anomalies**: When the trained model is given new, unseen data, it attempts to reconstruct it. If a sequence is anomalous (i.e., it doesn't fit the learned patterns), the model will struggle, resulting in a high reconstruction error. Any error that surpasses the pre-defined threshold is flagged as an anomaly.

## Files in This Project

*   `lstm_autoencoder.py`: The main Python script that loads the GE stock data, builds, trains, and evaluates the LSTM autoencoder.
*   `data/GE.csv`: The dataset containing historical daily stock prices for General Electric.
*   `basic.py`: A simpler example of an autoencoder using Dense layers on a synthetic dataset.
*   `data/anomaly.csv`: A synthetic dataset used by `basic.py` with pre-labeled "Good" and "Bad" data points.

## Requirements

This project requires the following Python libraries. You can install them using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

## How to Run

To run the main anomaly detection script, navigate to the directory containing the files and execute:

```bash
python lstm_autoencoder.py
```

This will run the script, display several plots for analysis during the process, and show the final anomaly detection results.

## Output Visualizations

The script generates a series of plots to help understand the model's behavior:

1.  **GE Stock Close Price Over Time**: The raw input data, showing the split between training and testing periods.
2.  **Model Loss**: Training and validation loss curves to check for model convergence and overfitting.
3.  **Training MAE Distribution**: A histogram of reconstruction errors on the normal data. This is used to set the anomaly threshold.
4.  **Test MAE Distribution**: A histogram of reconstruction errors on the test data. A wider spread indicates the presence of potential anomalies.
5.  **Anomaly Scores vs. Threshold**: A time-series plot showing the daily reconstruction error against the calculated anomaly threshold. Spikes above the threshold are flagged as anomalies.
6.  **Anomalies in GE Stock Price**: The final, conclusive plot showing the original stock price with detected anomalies highlighted in red.

## Key Hyperparameters

These important settings can be tuned at the top of `lstm_autoencoder.py` to adjust the model's behavior:

*   `SEQ_SIZE`: The length of the input sequences (e.g., `30` days). A longer sequence provides more temporal context for the model but increases computational cost.
*   `THRESHOLD_PERCENTILE`: The sensitivity of the detector. A value of `0.99` means any data with a reconstruction error in the top 1% of normal errors will be flagged as an anomaly.
    *   **Lowering this value** (e.g., to `0.95`) will make the detector more sensitive but may increase false positives.
    *   **Increasing this value** (e.g., to `0.995`) will make the detector less sensitive, flagging only the most extreme events.

## The `basic.py` Script

This script provides a much simpler introduction to the concept of autoencoders for anomaly detection.

*   **Data**: It uses the `anomaly.csv` file, which contains synthetic sensor readings (`Power`, `Detector`) and a `Quality` label.
*   **Model**: It uses a simple `Dense` layer autoencoder, as there is no time-series component.
*   **Logic**:
    1.  It trains the autoencoder *only* on the "Good" data.
    2.  It then calculates the reconstruction error (RMSE) for three groups: a test set of "Good" data, the full "Good" dataset, and the "Bad" data.
    3.  The script demonstrates that the reconstruction error for "Bad" data is significantly higher than for "Good" data, proving the concept.

