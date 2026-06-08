# Periodic Signal Model

## Overview

This project extends the TimesNet architecture for multivariate time-series forecasting by improving how periodic patterns are represented and combined over time.

The model first converts 1D time-series data into multiple 2D period-based representations using frequency analysis. These representations are processed with multi-scale convolutions to learn temporal patterns at different resolutions.

To capture changes in the strength of periodic components over time, the original architecture is extended with an amplitude-aware mechanism based on band-pass filtering and Hilbert transforms. The extracted amplitude information is then used in an adaptive aggregation mechanism that dynamically weights different periodic representations during prediction.

The approach is evaluated on the OmniAnomaly benchmark dataset for server telemetry forecasting and achieves improved forecasting performance compared to the baseline TimesNet model.

## Model Architecture

### 1D → 2D Transformation

Following the approach described above, the 1D time series data is transformed into  2D tensors that represent both intra- and interperiodic variations.
Specifically, a Fourier transform is applied to the input sequence to determine a fixed number of frequencies with the highest amplitudes. For each of these dominant frequencies, the time series is then segmented into intervals corresponding to the respective period length, which are arranged row‑wise. In this 2D representation, the rows capture variations within a single period, while the columns describe variations at identical phase positions across multiple periods.

### Convolutional Layer
The resulting 2D representations are then processed by a multi-scale convolutional block. To capture temporal patterns at different granularities, four parallel 2D convolutions with different kernel sizes are applied. Larger kernels capture broader dependencies across periods, while smaller kernels focus on local temporal structure. The outputs of all convolution branches are concatenated, projected back to the original model dimension, and reshaped into 1D time-series representations. This produces k period-specific sequences, which are subsequently processed by an MLP block before being aggregated.

### Instantaneous Amplitude Estimation
The architecture described in the TimesNet paper (https://arxiv.org/abs/2210.02186) selects dominant frequencies based on spectral amplitude, but does not model how amplitude varies over time. To adress this, I extended the model by introducing a filter block that captures variations of this kind. Specifically, for each dominant frequency, the original time series is passed through a band‑pass filter, resulting in a narrow‑band oscillatory component centered around the respective frequency. A Hilbert transform is then applied to this oscillatory mode to obtain the analytic signal, from which the instantaneous amplitude within the filter band can be derived. 

### Amplitude-Aware Adaptive Aggregation

After processing, the k period-specific time series are combined into a single output sequence using an adaptive weighting mechanism.

The aggregation consists of two components:

- **Global weights** are derived from the spectral amplitudes of the dominant frequency components identified during period detection. These weights determine the overall importance of each period-specific representation.
- **Local weights** are computed from the instantaneous amplitudes of each component through a learned projection layer. This enables the model to dynamically adjust the importance of individual time steps within each sequence.

A learnable scaling factor `alpha` controls the contribution of the local weights relative to the global weights. The final prediction is obtained as a weighted sum of the k processed time-series representations.

The aggregated sequence is passed to the next TimesBlock, where the period detection, convolutional processing, and aggregation steps are repeated. After all TimesBlocks have been processed, the final representation is projected back to the original feature dimension to produce the forecasted time series.

## Results

| Model                         | Val Loss (RMSE) | Δ        |
|-------------------------------|-----------------|----------|
| TimesNet (baseline)           | ~0.00079        | —        |
| + Amplitude-aware Aggregation | ~0.00066        | −16.5%   |
| + Hyperparameter Tuning (W&B) | ~0.00064        | −3%      |

Reported RMSE values correspond to forecasting with a context length of 128 and prediction length of 32.

The results show that the amplitude-aware extension consistently improves forecasting performance compared to the baseline TimesNet model, with further gains achieved through hyperparameter tuning.

The dataset is based on the [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) benchmark dataset, using the Machine-1-1 server telemetry subset for training and evaluation.

The default training configuration corresponds to the best-performing setup (amplitude-aware + hyperparameter-tuned model) and is integrated directly into the training pipeline.

The `experiments/` directory contains all experimental artifacts, including:
- a TimesNet baseline checkpoint
- a hyperparameter-tuned amplitude-aware checkpoint
- the training and evaluation dataset



## How to Use

### Training
The training script trains the model on the provided dataset and saves a trained checkpoint to disk. Hyperparameters can be modified via the configuration file `model/config.py`.
The output is a model checkpoint file containing the learned weights and model configuration.

```bash
python train.py \
  --train <path_to_train_dataset_file> \
  --val   <path_to_val_dataset_file>
```

Example:
```bash
python train.py \
  --train experiments/dataset/machine-1-1_train.txt \
  --val   experiments/dataset/machine-1-1_val.txt
```

### Inference
The inference script loads a trained model checkpoint and generates predictions on unseen data.

The output is a file containing the forecasted time series predictions.

```bash
python predict.py \
  --data <path_to_input_dataset_file> \
  --checkpoint <path_to_trained_model_checkpoint> \
  --output <path_to_output_file>
```

Example:
```bash
python predict.py \
  --data       experiments/dataset/machine-1-1_val.txt \
  --checkpoint experiments/checkpoints/amplitude_aware_tuned_checkpoint.pt \
  --output     results/predictions.csv
```


