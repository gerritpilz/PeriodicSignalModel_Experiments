# Periodic Signal Model

## Overview

Time series analysis plays a central role in a wide range of applications, where modeling complex temporal variations remains a fundamental challenge. Directly learning these variations from one‑dimensional sequences is difficult due to intricate temporal patterns. 

Motivated by the observation that many time series exhibit multiple underlying periodic structures, the temporal dynamics are decomposed into intra‑periodic variations (within a single peroid) and inter‑periodic variations (across periods at the same phase). To better expose these structures, the one‑dimensional time series is transformed into a set of two‑dimensional representations based on dominant periods, where rows and columns explicitly encode the two types of variation. This transformation embeds intra‑periodic and inter‑periodic variations into the columns and rows of the 2D tensors, respectively.

The resulting 2D representations are processed by a multi-scale convolutional block that captures both local and broader temporal dependencies before being projected back and adaptively aggregated across multiple period-specific representations.

The resulting one-dimensional sequence captures the relevant information of the original signal. In this work, the model is used for forecasting server machine time series data, focusing on accurate prediction of future system behaviour.

## Model Architecture

### 1D → 2D Transformation

Following the approach described above, the 1D time series data is transformed into  2D tensors that represent both intra- and interperiodic variations.
Specifically, a Fourier transform is applied to the input sequence to determine a fixed number of frequencies with the highest amplitudes. For each of these dominant frequencies, the time series is then segmented into intervals corresponding to the respective period length, which are arranged row‑wise. In this 2D representation, the rows capture variations within a single period, while the columns describe variations at identical phase positions across multiple periods.

## Convolutional Layer
The resulting 2D representations are then processed by a multi-scale convolutional block. To capture temporal patterns at different granularities, four parallel 2D convolutions with different kernel sizes are applied. Larger kernels capture broader dependencies across periods, while smaller kernels focus on local temporal structure. The outputs of all convolution branches are concatenated, projected back to the original model dimension, and reshaped into 1D time-series representations. This produces k period-specific sequences, which are subsequently processed by an MLP block before being aggregated.

## Instantaneous Amplitude Estimation
The architecture described in the TimesNet paper (https://arxiv.org/abs/2210.02186) selects dominant frequencies based on spectral amplitude, but does not model how amplitude varies over time. To adress this, I extended the model by introducing a filter block that captures variations of this kind. Specifically, for each dominant frequency, the original time series is passed through a band‑pass filter, resulting in a narrow‑band oscillatory component centered around the respective frequency. A Hilbert transform is then applied to this oscillatory mode to obtain the analytic signal, from which the instantaneous amplitude within the filter band can be derived. 

## Amplitude-aware Adaptive Aggregation

After processing, the k period-specific time series are combined into a single output sequence using an adaptive weighting mechanism.

The aggregation consists of two components:

- **Global weights** are derived from the spectral amplitudes of the dominant frequency components identified during period detection. These weights determine the overall importance of each period-specific representation.
- **Local weights** are computed from the instantaneous amplitudes of each component through a learned projection layer. This enables the model to dynamically adjust the importance of individual time steps within each sequence.

A learnable scaling factor `alpha` controls the contribution of the local weights relative to the global weights. The final prediction is obtained as a weighted sum of the k processed time-series representations.

The aggregated sequence is passed to the next TimesBlock, where the period detection, convolutional processing, and aggregation steps are repeated. After all TimesBlocks have been processed, the final representation is projected back to the original feature dimension to produce the forecasted time series.

## Results

| Model                         | Val Loss (RMSE) | Δ        |
|-------------------------------|-----------------|----------|
| TimesNet (baseline)           | ~X              | —        |
| + Amplitude-aware Aggregation | ~0.00066        | −X%      |
| + Hyperparameter Tuning (W&B) | ~X              | −Y%      |

The dataset is derived from the OmniAnomaly benchmark dataset (NetManAIOps), specifically using the Machine-1-1 subset. 

The default configuration in the training pipeline corresponds to the best-performing model (amplitude-aware + tuned setup) and is directly integrated into the codebase.
The `experiments/` directory contains all experimental artifacts, including:

- Baseline TimesNet model checkpoint  
- Amplitude-aware pre tuning model checkpoint  
- Hyperparameter-tuned amplitude-aware model checkpoint (best model)  
- Pre-tuning configuration files  
- The dataset used for training and evaluation

## How to Use

### Training
The training script trains the model on the provided dataset and saves a trained checkpoint to disk. Hyperparameters can be modified via the configuration file `config.py`.
The output is a model checkpoint file containing the learned weights and model configuration.

```bash
python train.py \
  --train <path_to_train_dataset_file> \
  --val   <path_to_val_dataset_file>
```

Example:
```bash
python train.py \
  --train dataset/machine-1-1_train.txt \
  --val   dataset/machine-1-1_val.txt
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
  --data       dataset/machine-1-1_val.txt \
  --checkpoint checkpoints/model.pt \
  --output     results/predictions.csv
```


