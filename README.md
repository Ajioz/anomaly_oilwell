# Sensor Anomaly Detection using an Autoencoder

This project demonstrates how to use an autoencoder neural network to detect anomalies in time-series sensor data from an industrial machine.

## Overview

The primary goal is to identify unusual patterns in sensor readings that may indicate a machine malfunction. We use an autoencoder, a type of unsupervised neural network, which is trained to reconstruct "normal" operational data. When the model is presented with data that deviates from this normal pattern (an anomaly), it will have a high reconstruction error, allowing us to flag it.

The process involves:
1.  **Data Exploration and Visualization**: Understanding the sensor data and its relationship with the machine's status.
2.  **Data Preprocessing**: Preparing the data for the model, including scaling and splitting.
3.  **Model Training**: Building and training an autoencoder on data representing only normal machine operation.
4.  **Anomaly Detection**: Establishing a threshold for reconstruction error and using it to identify anomalies in a test dataset that includes normal, broken, and recovering states.
5.  **Evaluation**: Visualizing the results to confirm the model's effectiveness.

## Dataset

The dataset used is `sensor.csv`, which contains time-series data from 51 sensors on a machine.

-   `timestamp`: The timestamp of the sensor reading.
-   `sensor_00` through `sensor_51`: The readings from the various sensors.
-   `machine_status`: The status of the machine at the time of the reading. This can be one of three states:
    -   **NORMAL**: The machine is operating as expected.
    -   **BROKEN**: The machine has failed.
    -   **RECOVERING**: The machine is in a transient state after a failure.

## Methodology

### 1. Data Preprocessing

The data is split into a training set and a test set based on time.

-   **Training Set**: Contains data exclusively from periods where `machine_status` is `NORMAL`. This is crucial because we want the autoencoder to learn what "normal" looks like.
-   **Test Set**: Contains data from all machine statuses to evaluate the model's ability to distinguish normal from anomalous behavior.

All sensor features are standardized using `StandardScaler` from scikit-learn. The scaler is fitted *only* on the normal training data to prevent information leakage from the anomalous data in the test set.

### 2. Autoencoder Architecture

A deep autoencoder is built using TensorFlow and Keras. The architecture is symmetrical:

-   **Encoder**: Compresses the input data (51 sensor features) into a lower-dimensional representation (latent space).
    -   Input (51) -> Dense(32, activation='relu') -> Dense(16, activation='relu') -> Dense(8, activation='relu')
-   **Decoder**: Attempts to reconstruct the original input data from the compressed representation.
    -   Latent Space (8) -> Dense(16, activation='relu') -> Dense(32, activation='relu') -> Output(51)

The model is compiled with the `adam` optimizer and uses `mean_absolute_error` (MAE) as the loss function.

### 3. Training and Thresholding

The autoencoder is trained on the preprocessed **normal** data. The goal is for the model to minimize the reconstruction error (the difference between the original input and the reconstructed output).

After training, we calculate the reconstruction loss (MAE) for each sample in the normal training data. A threshold for anomaly detection is then set based on the distribution of these errors. A common approach, used here, is:

`threshold = mean(training_loss) + 3 * std(training_loss)`

### 4. Anomaly Detection

To detect anomalies in the test set, we feed the test data through the trained autoencoder and calculate the reconstruction error for each data point.

-   If `reconstruction_error > threshold`, the data point is classified as an **anomaly**.
-   If `reconstruction_error <= threshold`, it is classified as **normal**.

## Results

The model successfully identifies the time periods where the machine was in a `BROKEN` state. The visualizations in the notebook clearly show that the reconstruction error spikes significantly during these periods, crossing the defined anomaly threshold. This confirms that the autoencoder effectively learned the patterns of normal operation and can distinguish them from faulty states.


*(Example image showing sensor readings with detected anomalies highlighted)*

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    The project requires the following Python libraries. You can install them using pip:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyterlab
    ```

4.  **Launch Jupyter and run the notebook:**
    ```bash
    jupyter lab
    ```
    Then, open and run the `sensor_anomaly_detection.ipynb` notebook.

## File Structure

```
├── sensor_anomaly_detection.ipynb  # The main Jupyter notebook with all the code and analysis.
├── sensor.csv                      # The dataset file.
└── README.md                       # This file.
```

This `README.md` provides a solid foundation for your project. You can add or modify sections as you see fit. For example, you might want to create a `requirements.txt` file for easier dependency management.
