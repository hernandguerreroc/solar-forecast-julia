# Solar Activity Forecasting with a Hybrid Conv1D‑LSTM Model  
## Validated on Solar Cycle 25

**Authors**  
Hernán Darío Guerrero‑Caguasango · Santiago Vargas‑Domínguez  
*Observatorio Astronómico Nacional, Universidad Nacional de Colombia, Bogotá, Colombia*

**Journal** *Solar Physics* (2026), **301**, Article 27105  
**DOI** [10.1007/s11207‑026‑02710‑5](https://doi.org/10.1007/s11207-026-02710-5)  
**Repository** Full source code to reproduce every result presented in the paper.

[![Julia](https://img.shields.io/badge/Julia-1.12.5-9558b2?logo=julia)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides a fully reproducible implementation in **Julia** of a hybrid **Conv1D‑LSTM** neural network for monthly sunspot number (SSN) forecasting. The model is trained on the entire SILSO v2.0 dataset (January 1749 – February 2026) and evaluated on the unseen Solar Cycles 24 and 25 using a strict one‑step‑ahead protocol that prevents any data leakage.

### Key methodological features

- **Physically motivated split** – Training on Cycles 1–23, validation on Cycle 24, and testing on Cycle 25.
- **Optimal window size** – Selected via the autocorrelation function (ACF) of the training series.
- **Automatic learning‑rate selection** – Geometric LR finder following Smith (2017).
- **Walk‑forward cross‑validation** – Performed on Cycles 20–23 to assess generalisation.
- **Morphological similarity** – Dynamic Time Warping (DTW) analysis to compare Cycle 25 with earlier cycles.
- **Post‑hoc uncertainty quantification** – Monte Carlo Dropout (Gal & Ghahramani, 2016) providing 90% credible intervals for the Cycle 26 projection.
- **Hardware‑aware execution** – Automatically detects NVIDIA CUDA + cuDNN, AMD ROCm, or falls back to multi‑threaded CPU.
- **Full reproducibility** – Fixed random seed (`Random.seed!(42)`) and a frozen package environment (`Manifest.toml`).

---

## Requirements

- **Julia** ≥ 1.9 (tested with 1.12.5, the version used in the paper).
- **Optional** – A compatible NVIDIA GPU (CUDA + cuDNN) or AMD GPU (ROCm). The script gracefully falls back to CPU if no supported GPU is found.

---

## Getting Started (Reproducing the Results)

Follow these steps to obtain exactly the same results as in the article:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Instantiate the exact package environment
This step reads Manifest.toml and installs the identical versions of every dependency used in the study.

bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
3. Run the main script
bash
julia --project=. --threads=auto solar_forecasting.jl
The --threads=auto flag enables multi‑threaded BLAS on CPU.

The first run will automatically download the SILSO dataset (≈ 3 MB) from the official server.

Data
The script downloads the data directly from the SILSO World Data Center (Royal Observatory of Belgium):

International Sunspot Number (ISN) v2.0, monthly mean total.

Source: https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv

No manual data preparation is required; the dataset is cached locally after the first download.

Outputs and Figures
After successful execution, the terminal will display:

Training/validation losses and MAEs.

One‑step‑ahead MAE and RMSE for Cycles 24 and 25.

The predicted peak of Solar Cycle 26 with a 90% credible interval.

The script generates publication‑quality vector PDFs (and accompanying PNGs) for all figures:

File name	Content
smoothing.pdf/.png	Raw monthly SSN and the 13‑month smoothed series (SILSO style).
spectrum.pdf/.png	Power spectrum with dominant periodicities (Schwabe, QBO, Hale).
acf.pdf/.png	Autocorrelation function and the selected window size.
dtw.pdf/.png	DTW distances between Cycle 25 and previous cycles.
lr_finder.pdf/.png	Learning‑rate finder curve with the optimal η* highlighted.
training_curve.pdf/.png	Loss and MAE curves for training and validation.
forecast.pdf/.png	One‑step‑ahead predictions for Cycles 24 and 25.
full_series.pdf/.png	Complete SSN record (1749–2036) with the Cycle 26 projection.
cycle26_uncertainty.pdf/.png	Cycle 26 mean trajectory and 90% MC Dropout credible band.
Repository Structure
text
.
├── Project.toml          # Direct dependencies (Flux, CUDA, cuDNN)
├── Manifest.toml         # Exact versions of all packages (reproducibility)
├── solar_prediction.jl   # Main source code
└── README.md             # This file
License
This project is distributed under the MIT License. You are free to use, modify, and redistribute the code, provided that appropriate credit is given to the original authors.

Citation
If you use this code in your research, please cite the associated article:

Guerrero‑Caguasango, H.D. and Vargas‑Domínguez, S., 2026. Solar Activity Forecasting Using a Hybrid Conv1D‑LSTM Model Validated on Solar Cycle 25. Solar Physics, 301, Art. no. 27105. DOI: 10.1007/s11207‑026‑02710‑5

Contact
For questions, suggestions, or collaboration inquiries, please contact the corresponding author:
hdguerreroc@unal.edu.co

Acknowledgements
The authors gratefully acknowledge the Observatorio Astronómico Nacional at the Universidad Nacional de Colombia for institutional support, and the SILSO World Data Center for providing the high‑quality sunspot data that made this work possible.
