# SDANet: A Stochastic Depth Attention Network for Unsupervised Seismic Denoising

[![](https://img.shields.io/badge/Dataset-v1.0-blue.svg)](https://github.com/aaspip/denoising-benchmark) ![](https://img.shields.io/badge/python-3.11-green.svg?style=flat)

**SDANet** is an unsupervised learning network designed to suppress uncorrelated noise in seismic data (e.g., high-amplitude erratic noise and incoherent random noise). By employing a symmetric feature-learning architecture, it breaks the black-box effect of neural networks through interpretable mechanisms and provides exceptional anti-noise capabilities. SDANet ensures high-fidelity amplitude reconstruction and structural continuity, preventing the artificial over-smoothing of subtle stratigraphic textures.

# :star: Overview

Current unsupervised denoising methods face a severe trade-off between over-parameterization (leading to overfitting on non-stationary outliers) and under-parameterization (causing over-smoothing). To address this, **SDANet** integrates the following core characteristics:
* **Symmetric U-Net Architecture**: Progressively reduces feature dimensions in the encoder and restores spatial details in the decoder with concatenation-based skip connections.
* **Multi-branch Self-Attention**: Captures global long-range waveform dependencies to effectively separate erratic noise from continuous reflections.
* **Dynamic Stochastic Depth Dropout (SDD)**: Adaptively scales the feature-dropping probability (e.g., from $p_1=0.0409$ in shallow layers to $p_6=0.1000$ in deep core layers) to disrupt feature co-adaptation and mitigate overfitting.
* **Hybrid Huber-TV Loss**: Couples the statistical robustness of Huber loss for high-amplitude anomalies with the edge-preserving constraints of Total Variation (TV) regularization.

## Workflow

- **Step 1:** The noisy seismic data undergoes a localized multidimensional decomposition. For 2D profiles, a $32 \times 32$ spatial-temporal window with a $1 \times 1$ stride is applied; for 3D volumetric data, a $12 \times 12 \times 12$ window with a $2 \times 2 \times 2$ stride is used to obtain $D_{patch}$.
- **Step 2:** All data $D_{patch}$ is split at a ratio of 8:2, with 80% allocated to the training set and the remaining 20% to the validation set.
- **Step 3:** The network is trained directly on noisy data. `DDIUL`, `PatchUNet`, `WMANet`, and `SDANet` are adopted to test and compare the denoising performance. The trained network parameters are saved as `.pth` files.
- **Step 4:** The visualization module is used to analyze patch data of specific features and observe the performance of different output layers, which provides direct evidence for the interpretability of the network's SDD probability allocation.

---

- `data`: All **data** is stored in this folder, including: `seis2dsyn.mat`, `seis3dsyn.mat`, `real2d.mat`, `real3d.mat`.

| Name | Data Size | Patch Size | Slid Size | 
| :--- | :--- | :--- | :--- |
| seis2dsyn | 502 $\times$ 64 | 32 $\times$ 32 | 1 $\times$ 1 |
| seis3dsyn | 126 $\times$ 32 $\times$ 32 | 12 $\times$ 12 $\times$ 12 | 2 $\times$ 2 $\times$ 2 | 
| real2d | 512 $\times$ 123 | 48 $\times$ 48 | 1 $\times$ 1 | 
| real3d | 256 $\times$ 64 $\times$ 48 | 12 $\times$ 12 $\times$ 12 | 4 $\times$ 2 $\times$ 2 | 

# :rocket: File Description

- `SDANet`: Both the denoising networks of SDANet for synthetic data and field data are stored here, including both 2D and 3D versions.
- `DDIUL`: The baseline denoising networks of DDIUL for performance comparison.
- `PatchUNet`: The baseline denoising networks of PatchUNet for performance comparison.
- `WMANet`: The baseline denoising networks of WMANet for performance comparison.
- `Visualization module`: `Visualization.ipynb` for the visualization of network parameters and SDD probability analysis.

:boom: **Note: Only one type of seismic data is presented here. If you are interested in other datasets, you can replace the corresponding input `.pth` model parameters and patch data `D_patch`.** # The network structure of the proposed method is shown below:
![SDANet Structure](https://github.com/lorens5/SDANet/blob/main/Fig/structure.png)

# :hotsprings: Example
- All data is stored in `.mat` format. You can run this script to convert the data from MATLAB to Python.
```makedown
import scipy.io as sio
data = sio.loadmat('your_data.mat')
Dc = data['DataClean']
Dn = data['DataNoisy']
```

## Taking synthetic 2D seismic data as an example
- The figure below compares the denoising results of DDIUL, PatchUNet, WMANet, and the proposed SDANet on a synthetic dataset with five linear events contaminated by random and erratic noise (initial SNR = -4.27 dB). SDANet achieves the highest SNR (14.10 dB) and preserves the continuity of seismic events with the cleanest background.

![Denoising comparison on synthetic 2D data](https://github.com/lorens5/SDANet/blob/main/Fig/syn1b.png)

- The f-k spectra further illustrate the energy concentration. SDANet shows the most compact energy distribution around the valid signals, closely matching the spectrum of the clean data, while other methods exhibit scattered background energy or spectral smearing.

![f-k spectra of denoised results](https://github.com/lorens5/SDANet/blob/main/Fig/syn1b2.png)

## Taking 3D field data as an example
- We test the methods on a 3D field dataset (Kerry, New Zealand). The figure below shows the denoised volumes and removed noise volumes of DDIUL, PatchUNet, WMANet, and SDANet. WMANet over-smoothes the horizons and leaks coherent reflections into the removed noise, while SDANet reduces complex field noise effectively without damaging valid structures.

![3D denoising results on field data](https://github.com/lorens5/SDANet/blob/main/Fig/real2a.png)

- 3D local similarity cubes confirm the signal preservation. DDIUL, PatchUNet, and WMANet display many high‑similarity regions (yellow/red), indicating wrongly removed or leaked reflections. SDANet shows a predominantly deep‑blue background, proving minimal signal leakage.

![3D local similarity cubes](https://github.com/lorens5/SDANet/blob/main/Fig/real2a1.png)

- A 2D slice extracted from the similarity cube along the inline direction highlights the advantage of SDANet. While the other methods exhibit obvious high‑similarity anomalies, SDANet retains a uniformly low similarity, demonstrating robust signal preservation.

![2D local similarity slices](https://github.com/lorens5/SDANet/blob/main/Fig/real2b.png)
```
```
# :sunrise_over_mountains: Maintainer
Chao Fu

For any questions regarding the dataset, `pth` or scripts, please open an issue in this repository.
