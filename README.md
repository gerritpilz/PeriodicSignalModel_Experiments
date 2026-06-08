# Periodic Signal Model

## Overview

Time series analysis plays a central role in a wide range of applications, where modeling complex temporal variations remains a fundamental challenge. Directly learning these variations from one‑dimensional sequences is difficult due to intricate temporal patterns. 

Motivated by the observation that many time series exhibit multiple underlying periodic structures, the temporal dynamics are decomposed into intra‑periodic variations (within a single peroid) and inter‑periodic variations (across periods at the same phase). To better expose these structures, the one‑dimensional time series is transformed into a set of two‑dimensional representations based on dominant periods, where rows and columns explicitly encode the two types of variation. This transformation embeds intra‑periodic and inter‑periodic variations into the columns and rows of the 2D tensors, respectively.

The resulting 2D representations are processed by a multi-scale convolutional block that captures both local and broader temporal dependencies before being projected back to the original model dimension.

The resulting one‑dimensional sequence captures the relevant information of the original signal. In this work, the model is applied to forecasting on server machine data, though it naturally extends to a range of other time series tasks, including classification and anomaly detection.

## Model Architecture

### 1D → 2D Transformation

Following the approach described above, the 1D time series data is transformed into  2D tensors that represent both intra- and interperiodic variations.
Specifically, a Fourier transform is applied to the input sequence to determine a fixed number of frequencies with the highest amplitudes. For each of these dominant frequencies, the time series is then segmented into intervals corresponding to the respective period length, which are arranged row‑wise. In this 2D representation, the rows capture variations within a single period, while the columns describe variations at identical phase positions across multiple periods.

## Convolutional Layer
The resulting 2D representations are then processed by a multi-scale convolutional block. To capture temporal patterns at different granularities, four parallel 2D convolutions with different kernel sizes are applied. Larger kernels capture broader dependencies across periods, while smaller kernels focus on local temporal structure. The outputs of all convolution branches are concatenated, projected back to the original model dimension, and reshaped into 1D time-series representations. This produces k period-specific sequences, which are subsequently processed by an MLP block before being aggregated.

## Amplitude Filter 
The architecture described in the TimesNet paper (https://arxiv.org/abs/2210.02186) selects dominant frequencies based on spectral amplitude, but does not model how amplitude varies over time. To adress this, I extended the model by introducing a filter block that captures variations of this kind. Specifically, for each dominant frequency, the original time series is passed through a band‑pass filter, resulting in a narrow‑band oscillatory component centered around the respective frequency. A Hilbert transform is then applied to this oscillatory mode to obtain the analytic signal, from which the instantaneous amplitude within the filter band can be derived. 

## Adaptive Aggregation

After processing, the k period-specific time series are combined into a single output sequence using an adaptive weighting mechanism.

The aggregation consists of two components:

- **Global weights** are derived from the spectral amplitudes of the dominant frequency components identified during period detection. These weights determine the overall importance of each period-specific representation.
- **Local weights** are computed from the instantaneous amplitudes of each component through a learned projection layer. This enables the model to dynamically adjust the importance of individual time steps within each sequence.

A learnable scaling factor `alpha` controls the contribution of the local weights relative to the global weights. The final prediction is obtained as a weighted sum of the k processed time-series representations.


The resulting sequence is fed into the next iteration of the block until all blocks are processed. The final output can then be used for downstream tasks, such as computing prediction logits.



## Tasks and Use Case

The proposed architecture is designed as a general representation learning framework for time series analysis. It supports a range of downstream tasks, including forecasting, imputation, classification, and anomaly detection.

In this work, the framework is applied to weather time series forecasting. Multiple variables are modeled simultaneously, allowing the joint prediction of channels such as temperature, humidity, and related meteorological signals.
