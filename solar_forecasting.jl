# ========================================================================================
# Solar Activity Forecasting Using a Hybrid Conv1D-LSTM Model Validated on Solar Cycle 25
# ========================================================================================
#
# Authors  : Hernán Darío Guerrero-Caguasango¹  (ORCID: 0000-0002-6657-8492)
#            Santiago Vargas-Domínguez¹
#
# Affil.   : ¹ Observatorio Astronómico Nacional
#              Universidad Nacional de Colombia
#              Bogotá, Colombia
#
# Contact  : hdguerreroc@unal.edu.co
#
# Version  : (March 2026)
# Language : Julia 1.9+
# License  : MIT
#
# Description
# -----------
# A reproducible hybrid Conv1D-LSTM pipeline for monthly SSN forecasting
# trained on the complete SILSO dataset (January 1749 – February 2026).
# Key methodological features:
#   1. Physically motivated split by solar cycle boundaries
#      (train: Cycles 1–23, validation: Cycle 24, test: Cycle 25)
#   2. Optimal window size determined from the autocorrelation function (ACF)
#   3. Automatic learning-rate selection via geometric LR finder (Smith 2017)
#   4. Rolling walk-forward cross-validation over Cycles 20–23
#   5. Dynamic Time Warping (DTW) for morphological cycle similarity analysis
#   6. 13-month SILSO smoothing and FFT spectral analysis
#   7. One-step-ahead forecast evaluation (no data leakage)
#   8. Early stopping with validation monitoring
#   9. Automatic CPU/GPU detection (NVIDIA CUDA, AMD ROCm, multi-thread CPU)
#  10. Full determinism via Random.seed!(42)
#
# Data source
# -----------
# SILSO World Data Center — Royal Observatory of Belgium
# International Sunspot Number (ISN) Version 2.0, monthly mean total
# URL: https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv
#
# Usage
# -----
#   julia --project=. --threads=auto solar_prediction.jl
#
# References
# ----------
# Hathaway (2015)           Living Reviews in Solar Physics 12:4
# Hochreiter & Schmidhuber  Neural Computation 9(8):1735 (1997)
# Pala & Atici (2019)       Solar Physics 294:50
# Smith (2017)              WACV 2017
# Berndt & Clifford (1994)  KDD Workshop
# Chang & Ide (2021)        JGR Solid Earth 126:e2021JB021991
# Bezanson et al. (2017)    SIAM Review 59(1):65
# =============================================================================

using CSV, DataFrames, Flux, Optimisers, Functors
using Flux: DataLoader
using Random, Statistics, Plots, Printf, Dates, Downloads
using StatsBase: autocor
using FFTW

Random.seed!(42)  # Fixed seed for full reproducibility
gr()              # GR backend — required for PDF vector output

# =============================================================================
# HARDWARE DETECTION
# Priority: NVIDIA (CUDA + cuDNN) > AMD (AMDGPU/ROCm) > CPU
# Falls back to multi-threaded CPU if no compatible GPU is found.
# =============================================================================
using LinearAlgebra

const GPU_BACKEND = begin
    local cuda_ok = false
    try
        using CUDA, cuDNN
        if CUDA.functional() && CUDA.has_cuda_gpu()
            # Validate cuDNN actually works with Conv1D at runtime.
            # Quadro K620 (CC 5.0) reports CUDA functional but fails
            # on cuDNN convolution with modern cuDNN versions.
            # A dry-run forward pass catches this before training starts.
            try
                _test_conv = Flux.Conv((4,), 1=>8, relu; pad=(3,0)) |> gpu
                _x_test    = CUDA.rand(Float32, 60, 1, 2)
                _test_conv(_x_test)   # triggers cuDNN — fails here on K620
                cuda_ok = true
            catch _cudnn_err
                @warn "cuDNN Conv1D test failed on " *
                      "$(CUDA.name(CUDA.device())) (CC too old?) — " *
                      "falling back to CPU."
                cuda_ok = false
            end
        end
    catch
    end

    local amd_ok = false
    if !cuda_ok
        try
            using AMDGPU
            amd_ok = AMDGPU.functional()
        catch
        end
    end

    cuda_ok ? :cuda : amd_ok ? :amd : :cpu
end

const USE_GPU = GPU_BACKEND != :cpu
const DEVICE  = USE_GPU ? gpu : cpu

if GPU_BACKEND == :cuda
    println("Hardware : NVIDIA GPU — $(CUDA.name(CUDA.device())) | CUDA $(CUDA.runtime_version())")
elseif GPU_BACKEND == :amd
    println("Hardware : AMD GPU — Flux + AMDGPU")
else
    n_blas = Sys.CPU_THREADS
    BLAS.set_num_threads(n_blas)
    println("Hardware : CPU — $n_blas BLAS threads active")
end

# =============================================================================
# CUSTOM LAYERS
# PermCWN  — permutes axes (W, C, B) → (C, W, B) between Conv1D and LSTM
# LastStep — extracts the final timestep from an LSTM sequence output
# Both layers are registered with Functors for GPU transfer compatibility.
# =============================================================================

struct PermCWN end
(::PermCWN)(x) = permutedims(x, (2, 1, 3))
Functors.@functor PermCWN

struct LastStep end
(::LastStep)(x::AbstractArray{T,3}) where T = x[:, end, :]
Functors.@functor LastStep

# =============================================================================
# HUBER LOSS
# L_δ(ŷ,y) = 0.5(ŷ-y)²         if |ŷ-y| < δ
#           = δ(|ŷ-y| - δ/2)    otherwise
# δ = 1.0 was chosen relative to the SSN statistics (mean≈81, std≈67):
# residuals < 1 unit are penalised quadratically; larger residuals linearly.
# This provides robustness to the anomalously high SSN values in early cycles.
# Reference: Huber (1964), Annals of Mathematical Statistics 35(1):73.
# =============================================================================

function huber_loss(ŷ, y; δ=1.0f0)
    r = abs.(ŷ .- y)
    mean(ifelse.(r .< δ, 0.5f0 .* r .^ 2, δ .* (r .- 0.5f0 .* δ)))
end

# =============================================================================
# DATA LOADING
# Downloads the SILSO ISN v2.0 monthly dataset on first run and caches it
# locally. Missing values (flagged as -1 in the source file) are replaced
# by the mean of their immediate left and right neighbours.
# =============================================================================

function load_data()
    dest = "SN_m_tot_V2.0.csv"
    if !isfile(dest)
        println("Downloading SILSO dataset...")
        Downloads.download(
            "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv", dest)
    else
        println("Using local dataset: $dest")
    end

    raw   = CSV.read(dest, DataFrame; delim=';', header=false)
    year  = Int.(raw[!, 1])
    month = Int.(raw[!, 2])
    sn    = Float32.(raw[!, 4])

    # Linear interpolation of missing observations (flagged as -1)
    for i in findall(sn .== -1f0)
        prev  = i > 1          ? sn[i-1] : 0f0
        next  = i < length(sn) ? sn[i+1] : prev
        sn[i] = (prev + next) / 2f0
    end

    dates = [Date(year[i], month[i], 1) for i in 1:length(year)]
    n     = length(sn)

    println("Dataset  : $(dates[1]) → $(dates[end])  ($n months)")
    (series=sn, dates=dates, n=n)
end

# =============================================================================
# 13-MONTH SILSO SMOOTHING
# Official SILSO weighting scheme (Hathaway 2015, Eq. 1):
#   S(t) = [0.5·x(t-6) + x(t-5) + ... + x(t+5) + 0.5·x(t+6)] / 12
# Half-weights at the two endpoint months reduce edge effects.
# Used ONLY for visualisation and spectral analysis. All model training
# and evaluation use the raw monthly series to preserve full variability.
# =============================================================================

function solar_smooth_13(series::Vector{Float32})
    n      = length(series)
    smooth = copy(series)
    for i in 7:n-6
        smooth[i] = (0.5f0*series[i-6] +
                     sum(series[i-5:i+5]) +
                     0.5f0*series[i+6]) / 12.0f0
    end
    # Boundary months: simple mean over available neighbours
    for i in vcat(1:6, n-5:n)
        smooth[i] = mean(series[max(1,i-6):min(n,i+6)])
    end
    smooth
end

# =============================================================================
# FFT SPECTRAL ANALYSIS
# Computes the one-sided power spectrum of the mean-centred SSN series.
# Identifies the five dominant periodicities in the range 24–400 months.
# Known solar periodicities:
#   Schwabe cycle    : ~132 months (~11 years)
#   Quasi-biennial   : ~26  months (~2.2 years)  [QBO]
#   Gleissberg cycle : ~960 months (~80 years)
# Sampling rate: 1 observation per month → Nyquist period = 2 months.
# =============================================================================

function spectral_analysis(series::Vector{Float32}, dates::Vector{Date})
    n         = length(series)
    s_centred = series .- mean(series)           # remove DC component
    spectrum  = abs.(fft(Float64.(s_centred)))
    half_n    = div(n, 2)
    power     = spectrum[1:half_n] .^ 2
    freqs     = collect(0:half_n-1) ./ n         # cycles per month
    periods   = vcat([Inf], 1.0 ./ freqs[2:end])

    # Five largest peaks in the physically meaningful range (24–400 months)
    valid_idx = findall(p -> 24 <= p <= 400, periods)
    top5_idx  = valid_idx[sortperm(power[valid_idx], rev=true)[1:min(5,length(valid_idx))]]

    println("\nDominant periodicities (FFT):")
    for idx in top5_idx
        @printf("  Period: %5.1f months (%4.1f years) | Relative power: %.2f\n",
                periods[idx], periods[idx]/12,
                power[idx]/maximum(power[valid_idx]))
    end

    periods, power, top5_idx
end

# =============================================================================
# SOLAR CYCLE PARTITION
# Physically motivated train/validation/test split aligned with official
# SILSO solar minimum dates (ISN v2.0, 13-month smoothed series).
#
# Official minimum dates (source: https://www.sidc.be/SILSO/cyclesmm):
#   Cycle 24 onset  : December 2008  (SSN = 2.2, deepest minimum in >100 yr)
#   Cycle 25 onset  : December 2019  (SSN = 1.8, confirmed September 2020)
#   Cycle 25 maximum: October  2024  (SSN = 160.9, confirmed STCE/SILSO)
#
# Partition (n = number of months):
#   Training   : Cycles 1–23  Jan 1749 – Nov 2008   n = 3,119
#   Validation : Cycle 24     Dec 2008 – Nov 2019   n =   132
#   Test       : Cycle 25     Dec 2019 – Feb 2026   n =    75
#
# Rationale: each Schwabe cycle is an independent physical unit defined by the
# polarity reversal of the solar magnetic dipole (Hale cycle = 2 Schwabe cycles).
# Evaluating on complete, unseen cycles eliminates the phase contamination that
# arises when a percentage-based split bisects an active cycle.
# =============================================================================

function solar_cycle_split(data)
    cycle24_start = Date(2008, 12, 1)   # Cycle 24 onset (SILSO official)
    cycle25_start = Date(2019, 12, 1)   # Cycle 25 onset (SILSO official)

    idx_c24 = findfirst(d -> d >= cycle24_start, data.dates)
    idx_c25 = findfirst(d -> d >= cycle25_start, data.dates)

    train = data.series[1:idx_c24-1]
    valid = data.series[idx_c24:idx_c25-1]
    test  = data.series[idx_c25:end]

    println("\nSolar cycle partition:")
    println("  Training   (Cycles  1–23) : $(length(train)) months  " *
            "($(data.dates[1]) → $(data.dates[idx_c24-1]))")
    println("  Validation (Cycle 24)     : $(length(valid)) months  " *
            "($(data.dates[idx_c24]) → $(data.dates[idx_c25-1]))")
    println("  Test       (Cycle 25)     : $(length(test)) months  " *
            "($(data.dates[idx_c25]) → $(data.dates[end]))")

    (train=train, valid=valid, test=test,
     idx_c24=idx_c24, idx_c25=idx_c25)
end

# =============================================================================
# OPTIMAL WINDOW SIZE VIA AUTOCORRELATION ANALYSIS
# The ACF of the SSN training series is computed for lags 1–200 months.
# The 95% confidence threshold under the null hypothesis of white noise is
# ±2/√N (Bartlett 1946), where N = 3,119 → threshold ≈ ±0.036.
# W is the smallest lag at which the ACF remains below this threshold for
# five consecutive months, rounded to the nearest 12-month multiple.
# Capped at 60 months to ensure ≥ 72 usable sliding-window observations
# in the validation set (Cycle 24: n = 132 months; 132 - 60 = 72 windows).
# Reference: Hathaway (2015), Living Reviews in Solar Physics 12:4.
# =============================================================================

function optimal_window_acf(series::Vector{Float32}; max_lag=200)
    n         = length(series)
    threshold = 2.0 / sqrt(n)           # 95% confidence bound
    acf_vals  = autocor(Float64.(series), 1:max_lag)

    # Find first sustained crossing below threshold (5 consecutive lags)
    window = max_lag
    for lag in 1:max_lag-5
        if all(abs.(acf_vals[lag:lag+4]) .< threshold)
            window = lag
            break
        end
    end

    # Round to nearest 12-month multiple; bound to [24, 60]
    window_rounded = clamp(round(Int, window / 12) * 12, 24, 60)

    println("\nACF window selection:")
    println("  95% confidence threshold : $(round(threshold; digits=4))")
    println("  First sustained ACF lag  : $window months")
    println("  Selected window W        : $window_rounded months")

    window_rounded
end

# =============================================================================
# DYNAMIC TIME WARPING (DTW)
# Computes the minimum-cost elastic alignment between two time series of
# potentially unequal length via dynamic programming (Berndt & Clifford 1994).
#
# Recurrence: D[i+1,j+1] = |x_i - y_j| + min(D[i,j+1], D[i+1,j], D[i,j])
# Final distance: D[n+1, m+1]
#
# Applied here to quantify morphological similarity between Solar Cycle 25
# and all prior cycles, drawing on the analogy with seismic waveform
# cross-correlation in earthquake relocation (Chang & Ide 2021;
# Hauksson & Shearer 2005).
# Used strictly for descriptive analysis — does not influence training
# or hyperparameter selection (no data leakage).
# =============================================================================

function dtw_distance(s1::Vector{Float32}, s2::Vector{Float32})
    n, m = length(s1), length(s2)
    D    = fill(Inf32, n+1, m+1)
    D[1, 1] = 0f0
    for i in 1:n, j in 1:m
        D[i+1, j+1] = abs(s1[i] - s2[j]) + min(D[i, j+1], D[i+1, j], D[i, j])
    end
    D[n+1, m+1]
end

function find_similar_cycles(data, splits)
    println("\nDTW morphological similarity to Solar Cycle 25:")

    cycle25 = splits.test   # reference series: Cycle 25 (test set)

    # Official SILSO solar minimum dates (source: sidc.be/SILSO/cyclesmm)
    minima = [
        Date(1755,  2, 1),   # C1  Feb 1755  SSN=14.0
        Date(1766,  6, 1),   # C2  Jun 1766  SSN=18.6
        Date(1775,  6, 1),   # C3  Jun 1775  SSN=12.0
        Date(1784,  9, 1),   # C4  Sep 1784  SSN=15.9
        Date(1798,  4, 1),   # C5  Apr 1798  SSN= 5.3  (Dalton minimum)
        Date(1810,  7, 1),   # C6  Jul 1810  SSN= 0.0  (Dalton minimum)
        Date(1823,  5, 1),   # C7  May 1823  SSN= 0.1
        Date(1833, 11, 1),   # C8  Nov 1833  SSN=12.2
        Date(1843,  7, 1),   # C9  Jul 1843  SSN=17.6
        Date(1855, 12, 1),   # C10 Dec 1855  SSN= 6.0
        Date(1867,  3, 1),   # C11 Mar 1867  SSN= 9.9
        Date(1878, 12, 1),   # C12 Dec 1878  SSN= 3.7
        Date(1890,  3, 1),   # C13 Mar 1890  SSN= 8.3
        Date(1902,  1, 1),   # C14 Jan 1902  SSN= 4.5
        Date(1913,  7, 1),   # C15 Jul 1913  SSN= 2.5
        Date(1923,  8, 1),   # C16 Aug 1923  SSN= 9.3
        Date(1933,  9, 1),   # C17 Sep 1933  SSN= 5.8
        Date(1944,  2, 1),   # C18 Feb 1944  SSN=12.9
        Date(1954,  4, 1),   # C19 Apr 1954  SSN= 5.1
        Date(1964, 10, 1),   # C20 Oct 1964  SSN=14.3
        Date(1976,  3, 1),   # C21 Mar 1976  SSN=17.8
        Date(1986,  9, 1),   # C22 Sep 1986  SSN=13.5
        Date(1996,  8, 1),   # C23 Aug 1996  SSN=11.2
        Date(2008, 12, 1),   # C24 Dec 2008  SSN= 2.2
        Date(2019, 12, 1),   # C25 Dec 2019  SSN= 1.8
    ]

    distances = Tuple{Int,Float32}[]
    for i in 1:length(minima)-1
        i_start = findfirst(d -> d >= minima[i],   data.dates)
        i_end   = findfirst(d -> d >= minima[i+1], data.dates)
        (isnothing(i_start) || isnothing(i_end)) && continue
        cycle = data.series[i_start:i_end-1]
        dist  = dtw_distance(cycle25, cycle)
        push!(distances, (i, dist))
    end

    sort!(distances, by=x->x[2])
    println("  Most similar cycles to Cycle 25 (smallest DTW distance):")
    for (rank, (cn, d)) in enumerate(distances[1:min(5,end)])
        @printf("  %d. Cycle %2d  |  DTW = %.1f\n", rank, cn, d)
    end

    distances
end

# =============================================================================
# SLIDING-WINDOW DATASET
# Creates N = len(series) - W overlapping windows of length W.
# X shape: (W, 1, N) — Conv1D input format in Flux (length, channels, batch)
# y shape: (1, N)    — scalar target = x(t + W)
# Optional shuffling randomises the order of windows in the training loader.
# =============================================================================

function windowed_dataset(series::Vector{Float32}, window_size, batch_size;
                          shuffle_buf=0)
    N = length(series) - window_size
    X = Array{Float32}(undef, window_size, 1, N)
    y = Array{Float32}(undef, 1, N)
    for i in 1:N
        X[:, 1, i] = series[i:i+window_size-1]
        y[1,    i] = series[i+window_size]
    end
    idxs = shuffle_buf > 0 ? shuffle(1:N) : collect(1:N)
    DataLoader((X[:,:,idxs], y[:,idxs]), batchsize=batch_size, shuffle=false)
end

# =============================================================================
# MODEL ARCHITECTURE  —  Conv1D + LSTM₁ + LSTM₂ + Dense₁ + Dense₂ + Dense₃
#
# Dimension flow (W = window size, B = batch size):
#   (W, 1,   B)  →  Conv1D(filters=132, kernel=4, ReLU, causal pad)
#   (W, 132, B)  →  PermCWN  (axis permutation)
#   (132, W, B)  →  LSTM(132→256, return_sequence=true)
#   (256, W, B)  →  LSTM(256→160, return_sequence=true)
#   (160, W, B)  →  LastStep  (select final timestep)
#   (160, B)     →  Dense(160→80,  ReLU)
#   (80,  B)     →  Dense(80→10,   ReLU)
#   (10,  B)     →  Dense(10→1)  × 400
#   (1,   B)     output
#
# Trainable parameters: 679,577
#   Conv1D  :     660   (132 × (4×1 + 1))
#   LSTM₁   : 398,336   (4 × (132×256 + 256×256 + 256))
#   LSTM₂   : 266,400   (4 × (256×160 + 160×160 + 160))
#   Dense₁  :  12,880   (160×80 + 80)
#   Dense₂  :     810   (80×10 + 10)
#   Dense₃  :      11   (10×1 + 1)
#
# Design notes:
#   - Causal padding (3,0) on Conv1D ensures each position depends only
#     on past inputs, preserving temporal ordering.
#   - LSTM₂ uses 160 units (= 5×32) following the GPU memory alignment
#     convention for multiples of 32 (Goodfellow et al. 2016), and provides
#     capacity to retain multiple SSN periodicities simultaneously.
#   - The ×400 output rescaling maps the model output to the SSN range
#     without normalising the training data, avoiding min-max instability
#     with the high-amplitude outliers present in early solar cycles.
# =============================================================================

function build_model(window_size)
    Chain(
        Conv((4,), 1=>132, relu; pad=(3,0)),   # causal Conv1D
        PermCWN(),                              # (W,C,B) → (C,W,B)
        LSTM(132=>256),                         # LSTM layer 1
        LSTM(256=>160),                         # LSTM layer 2
        LastStep(),                             # extract final timestep
        Dense(160=>80, relu),                   # fully connected 1
        Dense(80=>10,  relu),                   # fully connected 2
        Dense(10=>1),                           # output
        x -> x .* 400f0                         # rescale to SSN range
    ) |> DEVICE
end

# =============================================================================
# LEARNING-RATE FINDER
# Geometric schedule over `epochs` steps:
#   η(e) = base_lr × 10^(e/20),   e = 0, 1, ..., epochs-1
# Uses a freshly initialised model to avoid contaminating the main run.
# The epoch with minimum training loss is selected as η*.
# Reference: Smith (2017), WACV 2017.
# =============================================================================

function lr_finder(model, loader; epochs=100, base_lr=1f-8)
    println("\nLR Finder:")
    lrs    = [base_lr * 10f0^(Float32(e)/20f0) for e in 0:epochs-1]
    losses = Float32[]
    t0     = time()

    for (e, lr) in enumerate(lrs)
        Flux.reset!(model)
        opt_st  = Optimisers.setup(Optimisers.Momentum(lr, 0.9f0), model)
        ep_loss = 0f0;  nb = 0
        for (x, y) in loader
            x, y = x |> DEVICE, y |> DEVICE
            l, gs = Flux.withgradient(m -> huber_loss(m(x), y), model)
            Optimisers.update!(opt_st, model, gs[1])
            ep_loss += l;  nb += 1
        end
        push!(losses, ep_loss / nb)

        if e % 10 == 0
            eta = (time()-t0)/e * (epochs-e)
            @printf("  Epoch %3d/%d | loss=%.3f | ETA: %ds\n",
                    e, epochs, losses[end], round(Int, eta))
        end
    end

    best = argmin(losses)
    @printf("\n  Optimal LR: %.2e  (epoch %d, loss=%.3f)\n",
            lrs[best], best, losses[best])
    lrs[best], lrs, losses
end

# =============================================================================
# TRAINING WITH EARLY STOPPING
# Optimiser: SGD with Momentum (β = 0.9).
# Early stopping halts training when the validation Huber loss fails to
# improve for `patience` consecutive epochs; the model state from the
# epoch with the minimum validation loss is implicitly retained by the
# returned loss history (best_epoch is recorded for reference).
# =============================================================================

function train_model!(model, train_loader, valid_loader, epochs, lr;
                      patience=20)
    opt_st       = Optimisers.setup(Optimisers.Momentum(lr, 0.9f0), model)
    train_losses = Float32[];  train_maes = Float32[]
    valid_losses = Float32[];  valid_maes = Float32[]
    best_valid   = Inf32
    patience_cnt = 0
    best_epoch   = 0
    t0           = time()

    println("\n  Training (early stopping, patience=$patience):")
    for epoch in 1:epochs

        # Training step
        Flux.reset!(model);  Flux.trainmode!(model)
        ep_loss = 0f0;  ep_mae = 0f0;  nb = 0
        for (x, y) in train_loader
            x, y = x |> DEVICE, y |> DEVICE
            l, gs = Flux.withgradient(model) do m
                huber_loss(m(x), y)
            end
            Optimisers.update!(opt_st, model, gs[1])
            ep_loss += l;  ep_mae += mean(abs.(model(x) .- y));  nb += 1
        end
        push!(train_losses, ep_loss/nb);  push!(train_maes, ep_mae/nb)

        # Validation step
        Flux.reset!(model);  Flux.testmode!(model)
        vl = 0f0;  vm = 0f0;  vb = 0
        for (x, y) in valid_loader
            x, y = x |> DEVICE, y |> DEVICE
            ŷ = model(x)
            vl += huber_loss(ŷ, y);  vm += mean(abs.(ŷ .- y));  vb += 1
        end
        push!(valid_losses, vl/vb);  push!(valid_maes, vm/vb)

        # Early stopping check
        if valid_losses[end] < best_valid
            best_valid   = valid_losses[end]
            best_epoch   = epoch
            patience_cnt = 0
        else
            patience_cnt += 1
        end

        if epoch % 20 == 0
            eta = (time()-t0)/epoch * (epochs-epoch)
            @printf("  Epoch %3d/%d | Train=%.3f | Valid=%.3f | ETA: %ds\n",
                    epoch, epochs,
                    train_losses[end], valid_losses[end],
                    round(Int, eta))
        end

        if patience_cnt >= patience
            println("\n  Early stopping at epoch $epoch " *
                    "(best epoch: $best_epoch, valid loss: " *
                    "$(round(best_valid; digits=3)))")
            break
        end
    end
    train_losses, valid_losses, train_maes, valid_maes, best_epoch
end

# =============================================================================
# ROLLING WALK-FORWARD CROSS-VALIDATION
# Evaluates model generalisation over four consecutive solar cycles (20–23).
# At each fold, the model is trained from scratch on all available data
# up to cycle N and evaluated on cycle N+1 using the one-step-ahead protocol.
# Reference: Tashman (2000), International Journal of Forecasting 16(4):437.
# =============================================================================

function rolling_cv(data, window_size, batch_size, lr; epochs=100)
    println("\nRolling Walk-Forward Cross-Validation:")

    # Official SILSO minimum dates for fold boundaries
    minima = [
        Date(1964, 10, 1),   # Cycle 20 onset  Oct 1964  SSN=14.3
        Date(1976,  3, 1),   # Cycle 21 onset  Mar 1976  SSN=17.8
        Date(1986,  9, 1),   # Cycle 22 onset  Sep 1986  SSN=13.5
        Date(1996,  8, 1),   # Cycle 23 onset  Aug 1996  SSN=11.2
        Date(2008, 12, 1),   # Cycle 24 onset  Dec 2008  SSN= 2.2
    ]

    cv_maes = Float32[]

    for fold in 1:length(minima)-1
        i_train_end = findfirst(d -> d >= minima[fold],   data.dates) - 1
        i_valid_end = findfirst(d -> d >= minima[fold+1], data.dates) - 1

        train_s = data.series[1:i_train_end]
        valid_s = data.series[i_train_end+1:i_valid_end]

        # Adaptive batch size: never exceed available windows
        n_valid_w = length(valid_s) - window_size
        vbatch    = min(batch_size, n_valid_w)
        tr_loader = windowed_dataset(train_s, window_size, batch_size;
                                     shuffle_buf=900)
        vl_loader = windowed_dataset(valid_s, window_size, vbatch)

        m = build_model(window_size)
        train_model!(m, tr_loader, vl_loader, epochs, lr; patience=15)

        # One-step-ahead evaluation
        tv_series = vcat(train_s, valid_s)
        n_tr      = length(train_s)
        preds     = Float32[]
        Flux.reset!(m);  Flux.testmode!(m)
        for i in 1:length(valid_s)
            x = reshape(tv_series[n_tr+i-window_size:n_tr+i-1],
                        window_size, 1, 1) |> DEVICE
            ŷ = USE_GPU ? Array(m(x))[1] : m(x)[1]
            push!(preds, ŷ)
        end
        mae          = mean(abs.(preds .- valid_s))
        valid_cycle  = fold + 19   # Cycle numbers: 20, 21, 22, 23
        push!(cv_maes, mae)
        @printf("  Fold %d | Train to %s | Cycle %d | MAE=%.3f\n",
                fold, data.dates[i_train_end], valid_cycle, mae)
    end

    @printf("\n  CV MAE : %.3f ± %.3f\n", mean(cv_maes), std(cv_maes))
    cv_maes
end

# =============================================================================
# AUTOREGRESSIVE MULTI-STEP FORECAST
# Iteratively feeds each predicted value back as input for the next step.
# Used ONLY for the prospective Solar Cycle 26 projection (120 months
# beyond February 2026). Not used for Cycle 24 or Cycle 25 evaluation,
# which use the one-step-ahead protocol to avoid error accumulation.
# =============================================================================

function forecast_future(model, series::Vector{Float32}, window_size, steps)
    Flux.reset!(model);  Flux.testmode!(model)
    window = copy(series[end-window_size+1:end])
    preds  = Float32[]
    for _ in 1:steps
        x    = reshape(window, window_size, 1, 1) |> DEVICE
        pred = USE_GPU ? Array(model(x))[1] : model(x)[1]
        push!(preds, pred)
        window = vcat(window[2:end], pred)
    end
    preds
end

# =============================================================================
# MONTE CARLO DROPOUT — POST-HOC UNCERTAINTY ESTIMATION
#
# Strategy B (Gal & Ghahramani 2016): apply MC Dropout to the already-trained
# model WITHOUT retraining. The trained weights are preserved exactly; Dropout
# is injected as a functional wrapper only during the stochastic passes for
# Cycle 26. This recovers the original evaluation metrics (no degradation from
# Dropout regularisation) while providing meaningful uncertainty intervals.
#
# Scientific justification:
#   Gal & Ghahramani (2016) showed that applying Dropout at test time to a
#   pre-trained network approximates variational Bayesian inference, even when
#   the model was not trained with Dropout. The resulting intervals represent
#   epistemic (model) uncertainty — the uncertainty due to limited training
#   data and imperfect model capacity, distinct from aleatoric (observational)
#   uncertainty in the SSN series itself.
#
# Implementation:
#   For each of n_passes stochastic samples, a temporary chain wraps the
#   trained model layers with Dropout inserted before each Dense hidden layer.
#   trainmode! activates the Dropout; testmode! on the original model ensures
#   one-step-ahead evaluation is never contaminated.
#
# Parameters:
#   p_drop   : Dropout probability (default 0.10 — conservative for post-hoc)
#   n_passes : number of stochastic forward passes (200 standard)
# =============================================================================

function mc_dropout_posthoc(model, series::Vector{Float32},
                             window_size, steps;
                             p_drop=0.10f0, n_passes=200)
    println("  MC Dropout: $n_passes stochastic passes...")

    # Build a temporary chain that wraps trained layers with Dropout injected.
    # The trained parameters are shared (no copy) — only the forward-pass
    # topology changes by inserting Dropout nodes.
    # Layer order in build_model: Conv, PermCWN, LSTM1, LSTM2, LastStep,
    #                             Dense1(relu), Dense2(relu), Dense3, ×400
    layers = model.layers
    mc_model = Chain(
        layers[1],              # Conv1D (trained)
        layers[2],              # PermCWN
        layers[3],              # LSTM1  (trained)
        layers[4],              # LSTM2  (trained)
        layers[5],              # LastStep
        Dropout(p_drop),        # injected — active in trainmode
        layers[6],              # Dense(160→80, relu) (trained)
        Dropout(p_drop),        # injected — active in trainmode
        layers[7],              # Dense(80→10, relu)  (trained)
        layers[8],              # Dense(10→1)         (trained)
        layers[9]               # ×400 scaling
    ) |> DEVICE

    all_preds = Matrix{Float32}(undef, n_passes, steps)

    for s in 1:n_passes
        Flux.reset!(mc_model)
        Flux.trainmode!(mc_model)   # activates injected Dropout layers
        window = copy(series[end-window_size+1:end])
        for t in 1:steps
            x    = reshape(window, window_size, 1, 1) |> DEVICE
            pred = USE_GPU ? Array(mc_model(x))[1] : mc_model(x)[1]
            all_preds[s, t] = pred
            window = vcat(window[2:end], Float32(pred))
        end
        s % 50 == 0 && @printf("    Pass %d/%d
", s, n_passes)
    end

    # Summary statistics across stochastic passes
    mean_pred = Float32.(vec(mean(all_preds, dims=1)))
    p05_pred  = Float32.(vec(mapslices(
                    v -> quantile(v, 0.05f0), all_preds, dims=1)))
    p95_pred  = Float32.(vec(mapslices(
                    v -> quantile(v, 0.95f0), all_preds, dims=1)))

    # Report interval width at key horizons
    for t in [12, 24, 60, 120]
        t <= steps && @printf(
            "  90%% CI width at t=%3d months: %.1f SSN
",
            t, p95_pred[t] - p05_pred[t])
    end
    pk = argmax(mean_pred)
    @printf("  C26 peak estimate: %.1f SSN  (90%% CI: [%.1f, %.1f])  ~%d
",
            mean_pred[pk], p05_pred[pk], p95_pred[pk],
            Dates.year(series isa Vector ? Date(2026,3,1) + Month(pk) : Date(2026,3,1)))

    mean_pred, p05_pred, p95_pred
end

# =============================================================================
# EVALUATION METRICS
# =============================================================================

function compute_metrics(actual, predicted)
    mae  = mean(abs.(predicted .- actual))
    rmse = sqrt(mean((predicted .- actual).^2))
    mae, rmse
end

# =============================================================================
# FIGURES  —  Publication-quality vector PDF output
# All figures are saved as both PDF (vector, for the paper) and PNG (raster,
# for quick inspection). Styling follows Solar Physics journal conventions.
# =============================================================================

function plot_smoothing(dates, raw, smoothed)
    p = plot(dates, raw,
        lw=0.7, color=:lightblue, alpha=0.6,
        label="Monthly SSN (raw)",
        xlabel="Date",
        ylabel="Sunspot Number (SSN)",
        title="Monthly SSN and 13-Month Smoothed Series (SILSO, 1749–2026)",
        legend=:topleft, grid=true, gridalpha=0.25,
        size=(1400, 420), dpi=300,
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    plot!(p, dates, smoothed,
        lw=2.0, color=:royalblue,
        label="13-month smoothed (SILSO)")
    savefig(p, "smoothing.pdf");  savefig(p, "smoothing.png");  display(p)
end

function plot_spectrum(periods, power, top5_idx)
    valid = findall(p -> 12 <= p <= 400, periods)
    pmax  = maximum(power[valid])
    norm  = power[valid] ./ pmax   # normalise to [0, 1]

    p = plot(periods[valid], norm,
        lw=1.8, color=:steelblue,
        xlabel="Period (months)",
        ylabel="Normalised power spectral density",
        title="SSN Power Spectrum (1749–2026)",
        titlefontsize=13,
        legend=:topright, grid=true, gridalpha=0.25,
        size=(960, 540), dpi=300,
        label="Spectrum",
        left_margin=10Plots.mm, bottom_margin=8Plots.mm,
        top_margin=6Plots.mm)

    # Stagger annotation heights to avoid overlap with reference lines
    sorted_idx = sort(top5_idx, by=x->periods[x])
    y_offsets  = [0.10, 0.07, 0.13, 0.08, 0.11]
    for (k, idx) in enumerate(sorted_idx)
        per = periods[idx]
        if 12 <= per <= 400
            npower = power[idx] / pmax
            scatter!(p, [per], [npower], color=:crimson, markersize=7, label="")
            annotate!(p, per, min(npower + y_offsets[k], 0.97),
                text("$(round(per/12, digits=1))yr", :crimson, :center, 9))
        end
    end
    vline!(p, [132.0], color=:darkorange, linestyle=:dash, lw=1.8,
        label="Schwabe cycle (~11yr)")
    vline!(p, [26.0],  color=:darkgreen,  linestyle=:dash, lw=1.8,
        label="QBO (~2.2yr)")
    vline!(p, [264.0], color=:purple,     linestyle=:dot,  lw=1.5,
        label="Hale cycle (~22yr)")
    savefig(p, "spectrum.pdf");  savefig(p, "spectrum.png");  display(p)
end

function plot_acf(series, window_size; max_lag=200)
    acf_vals  = autocor(Float64.(series), 1:max_lag)
    threshold = 2.0 / sqrt(length(series))
    p = plot(1:max_lag, acf_vals,
        lw=1.5, color=:royalblue,
        xlabel="Lag (months)",
        ylabel="ACF",
        title="Autocorrelation Function — Training Series (Cycles 1–23)",
        legend=:topright, grid=true, gridalpha=0.25,
        size=(780, 460), dpi=300,
        label="ACF",
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    hline!(p, [threshold, -threshold],
        color=:crimson, linestyle=:dash, lw=1.0,
        label="95% CI  (±2/√N,  N=$(length(series)))")
    vline!(p, [window_size],
        color=:darkorange, linestyle=:dot, lw=2.0,
        label="Selected window  W = $window_size months")
    savefig(p, "acf.pdf");  savefig(p, "acf.png");  display(p)
end

function plot_dtw(distances)
    sorted = sort(distances, by=x->x[1])
    cycles = [d[1] for d in sorted]
    dists  = [d[2] for d in sorted]
    p = bar(cycles, dists,
        color=:steelblue, linecolor=:white, lw=0.5,
        xlabel="Solar cycle number",
        ylabel="DTW distance (SSN units)",
        title="Morphological Similarity to Solar Cycle 25 (DTW)",
        legend=false, grid=true, gridalpha=0.25,
        size=(780, 460), dpi=300,
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(p, "dtw.pdf");  savefig(p, "dtw.png");  display(p)
end

function plot_lr_finder(lrs, losses, opt_lr)
    idx = argmin(losses)
    p = plot(lrs, losses,
        xscale=:log10, lw=2.0, color=:steelblue,
        xlabel="Learning rate",
        ylabel="Huber loss",
        title="Learning-Rate Finder",
        legend=:topright, grid=true, gridalpha=0.25,
        size=(700, 460), dpi=300,
        label="Training loss",
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    scatter!(p, [opt_lr], [losses[idx]],
        color=:crimson, markersize=8,
        label="Optimal  η* = $(round(opt_lr; sigdigits=2))")
    savefig(p, "lr_finder.pdf");  savefig(p, "lr_finder.png");  display(p)
end

function plot_training(train_losses, valid_losses, train_maes, valid_maes,
                       best_epoch)
    p = plot(train_losses,
        lw=2.0, color=:royalblue,
        label="Train — Huber loss",
        xlabel="Epoch",
        ylabel="Loss / MAE",
        title="Training Curves  (best epoch: $best_epoch)",
        legend=:topright, grid=true, gridalpha=0.25,
        size=(720, 480), dpi=300,
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    plot!(p, valid_losses,
        lw=2.0, color=:darkorange,   label="Valid — Huber loss  (Cycle 24)")
    plot!(p, train_maes,
        lw=1.5, color=:royalblue,    linestyle=:dash, label="Train — MAE")
    plot!(p, valid_maes,
        lw=1.5, color=:darkorange,   linestyle=:dash, label="Valid — MAE")
    vline!(p, [best_epoch],
        color=:gray, linestyle=:dot, lw=1.5, label="Best epoch")
    savefig(p, "training_curve.pdf"); savefig(p, "training_curve.png"); display(p)
end

function plot_forecast_cycles(data, splits, forecast_valid, forecast_test,
                               val_mae, val_rmse, test_mae, test_rmse)
    dates_valid = data.dates[splits.idx_c24:splits.idx_c25-1]
    dates_test  = data.dates[splits.idx_c25:end]

    p = plot(dates_valid, splits.valid,
        lw=2.0, color=:royalblue,
        label="Cycle 24 — observed",
        xlabel="Date",
        ylabel="Sunspot Number (SSN)",
        title="One-Step-Ahead Forecast by Solar Cycle\n" *
              "Validation (C24): MAE=$(round(val_mae;digits=1))  " *
              "RMSE=$(round(val_rmse;digits=1))  |  " *
              "Test (C25): MAE=$(round(test_mae;digits=1))  " *
              "RMSE=$(round(test_rmse;digits=1))",
        legend=:topleft, grid=true, gridalpha=0.25,
        size=(1000, 520), dpi=300,
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    plot!(p, dates_valid, forecast_valid,
        lw=2.0, color=:darkorange, linestyle=:dash,
        label="Cycle 24 — predicted")
    plot!(p, dates_test, splits.test,
        lw=2.0, color=:steelblue,
        label="Cycle 25 — observed")
    plot!(p, dates_test, forecast_test,
        lw=2.0, color=:crimson, linestyle=:dash,
        label="Cycle 25 — predicted")
    savefig(p, "forecast.pdf");  savefig(p, "forecast.png");  display(p)
end

function plot_full_series(data, splits, forecast_valid, forecast_test,
                          future_dates, future_preds)
    p = plot(data.dates[1:splits.idx_c24-1],
             data.series[1:splits.idx_c24-1],
        lw=1.0, color=:royalblue, alpha=0.85,
        label="Training set (Cycles 1–23)",
        xlabel="Date",
        ylabel="Sunspot Number (SSN)",
        title="SSN 1749–$(year(future_dates[end])) | Conv1D-LSTM Forecast  " *
              "(Guerrero-Caguasango & Vargas-Domínguez 2026)",
        legend=:topleft, grid=true, gridalpha=0.25,
        size=(1400, 480), dpi=300,
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    plot!(p, data.dates[splits.idx_c24:splits.idx_c25-1], splits.valid,
        lw=1.5, color=:steelblue,   label="Cycle 24 — observed")
    plot!(p, data.dates[splits.idx_c24:splits.idx_c25-1], forecast_valid,
        lw=2.0, color=:darkorange,  linestyle=:dash,
        label="Cycle 24 — predicted")
    plot!(p, data.dates[splits.idx_c25:end], splits.test,
        lw=1.5, color=:steelblue,   label="")
    plot!(p, data.dates[splits.idx_c25:end], forecast_test,
        lw=2.0, color=:crimson,     linestyle=:dash,
        label="Cycle 25 — predicted")
    plot!(p, future_dates, future_preds,
        lw=2.0, color=:purple,      linestyle=:dot,
        label="Cycle 26 — projection (autoregressive, illustrative)")
    vline!(p, [data.dates[splits.idx_c24]],
        color=:gray,     linestyle=:dot, lw=1.2, label="Cycle 24 onset")
    vline!(p, [data.dates[splits.idx_c25]],
        color=:darkgray, linestyle=:dot, lw=1.2, label="Cycle 25 onset")
    savefig(p, "full_series.pdf");  savefig(p, "full_series.png");  display(p)
end

# =============================================================================
# FIGURE: CYCLE 26 PROJECTION WITH MC DROPOUT UNCERTAINTY BAND
# The 90% credible interval from MC Dropout post-hoc inference.
# Widening with horizon reflects compounding autoregressive uncertainty,
# consistent with the ~2yr predictability limit of the solar dynamo
# (Petrovay 2010, Living Reviews Solar Phys. 12:4).
# =============================================================================

function plot_cycle26_uncertainty(data, mean_pred, p05_pred, p95_pred, future_dates)
    # Show last 36 months of Cycle 25 as context
    n_ctx      = 36
    ctx_dates  = data.dates[end-n_ctx+1:end]
    ctx_series = data.series[end-n_ctx+1:end]

    p = plot(ctx_dates, ctx_series,
        lw=2.0, color=:steelblue,
        label="Cycle 25 — observed (last 36 months)",
        xlabel="Date",
        ylabel="Sunspot Number (SSN)",
        title="Solar Cycle 26 Projection — MC Dropout 90% Credible Interval
" *
              "(post-hoc, p=0.10, N=200 stochastic passes; " *
              "Gal & Ghahramani 2016)",
        titlefontsize=10,
        legend=:topright, grid=true, gridalpha=0.25,
        size=(1100, 500), dpi=300,
        left_margin=10Plots.mm, bottom_margin=8Plots.mm,
        top_margin=6Plots.mm)

    # 90% credible band
    plot!(p, future_dates, p05_pred,
        fillrange=p95_pred, fillalpha=0.18, color=:purple,
        lw=0.0, label="90% credible interval")
    plot!(p, future_dates, p05_pred,
        lw=0.8, color=:purple, linestyle=:dot, label="")
    plot!(p, future_dates, p95_pred,
        lw=0.8, color=:purple, linestyle=:dot, label="")

    # MC mean
    plot!(p, future_dates, mean_pred,
        lw=2.5, color=:purple, label="Cycle 26 — MC Dropout mean")

    # Mark projection start
    vline!(p, [data.dates[end]],
        color=:gray, linestyle=:dash, lw=1.2,
        label="Start of projection (Feb 2026)")

    savefig(p, "cycle26_uncertainty.pdf")
    savefig(p, "cycle26_uncertainty.png")
    display(p)
end

# =============================================================================
# MAIN PIPELINE
# =============================================================================

function main()

    # ── Hyperparameters ─────────────────────────────────────────────────────
    batch_size   = 145     # mini-batch size (training DataLoader)
    shuffle_buf  = 900     # window shuffle buffer (training only)
    lr_epochs    = 100     # epochs for LR finder
    train_epochs = 300     # maximum training epochs (early stopping may halt)
    future_steps = 120     # autoregressive projection horizon (months, ~10yr)
    mc_n_passes  = 200     # MC Dropout stochastic passes for C26 uncertainty
    run_cv       = true    # set to false to skip rolling cross-validation

    println("=" ^ 62)
    println("  Solar Sunspot Number Forecasting — Conv1D-LSTM  v2.0")
    println("  Observatorio Astronómico Nacional")
    println("  Universidad Nacional de Colombia")
    println("=" ^ 62)

    # ── Data loading ─────────────────────────────────────────────────────────
    println("\n[1] Loading SILSO dataset...")
    data = load_data()

    # ── Smoothing and spectral analysis ──────────────────────────────────────
    println("\n[2] 13-month smoothing and spectral analysis...")
    smoothed = solar_smooth_13(data.series)
    plot_smoothing(data.dates, data.series, smoothed)
    periods, power, top5_idx = spectral_analysis(data.series, data.dates)
    plot_spectrum(periods, power, top5_idx)

    # ── Solar cycle partition ─────────────────────────────────────────────────
    println("\n[3] Solar cycle partition...")
    splits = solar_cycle_split(data)

    # ── Optimal window size (ACF) ─────────────────────────────────────────────
    println("\n[4] Optimal window size via ACF...")
    window_size = optimal_window_acf(splits.train)
    plot_acf(splits.train, window_size)

    # ── DTW morphological analysis ────────────────────────────────────────────
    println("\n[5] DTW morphological analysis...")
    dtw_distances = find_similar_cycles(data, splits)
    plot_dtw(dtw_distances)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    n_valid_w   = length(splits.valid) - window_size
    n_test_w    = length(splits.test)  - window_size
    valid_batch = min(batch_size, n_valid_w)
    test_batch  = min(batch_size, n_test_w)

    println("\n[6] Sliding-window datasets:")
    println("  Train windows : $(length(splits.train) - window_size)")
    println("  Valid windows : $n_valid_w  (batch = $valid_batch)")
    println("  Test  windows : $n_test_w  (batch = $test_batch)")

    train_loader = windowed_dataset(splits.train, window_size, batch_size;
                                    shuffle_buf=shuffle_buf)
    valid_loader = windowed_dataset(splits.valid, window_size, valid_batch)

    (xb, yb) = first(train_loader)
    println("  Batch X shape : $(size(xb))  |  Batch y shape : $(size(yb))")

    # ── Learning-rate finder ──────────────────────────────────────────────────
    println("\n[7] Learning-rate finder ($lr_epochs epochs)...")
    temp_model = build_model(window_size)
    opt_lr, lrs, lr_losses = lr_finder(temp_model, train_loader;
                                        epochs=lr_epochs)
    plot_lr_finder(lrs, lr_losses, opt_lr)

    # ── Rolling cross-validation ──────────────────────────────────────────────
    if run_cv
        println("\n[8] Rolling walk-forward cross-validation...")
        cv_maes = rolling_cv(data, window_size, batch_size,
                              opt_lr; epochs=100)
    end

    # ── Main training run ─────────────────────────────────────────────────────
    println("\n[9] Training (max $train_epochs epochs, " *
            "lr=$(round(opt_lr; sigdigits=3)))...")
    model = build_model(window_size)
    train_losses, valid_losses, train_maes, valid_maes, best_epoch =
        train_model!(model, train_loader, valid_loader,
                     train_epochs, opt_lr; patience=20)
    plot_training(train_losses, valid_losses, train_maes, valid_maes, best_epoch)

    # ── One-step-ahead forecast: Cycle 24 (validation) ───────────────────────
    # For each month t in Cycle 24, the model receives the W=60 real
    # monthly observations immediately preceding t and predicts x(t).
    # No data leakage: ŷ(t) uses only data observed up to t-1.
    println("\n[10] One-step-ahead forecast: Cycle 24 (validation)...")
    series_tv      = vcat(splits.train, splits.valid)
    n_train        = length(splits.train)
    forecast_valid = Float32[]
    Flux.reset!(model);  Flux.testmode!(model)
    for i in 1:length(splits.valid)
        x = reshape(series_tv[n_train+i-window_size:n_train+i-1],
                    window_size, 1, 1) |> DEVICE
        ŷ = USE_GPU ? Array(model(x))[1] : model(x)[1]
        push!(forecast_valid, ŷ)
    end
    val_mae, val_rmse = compute_metrics(splits.valid, forecast_valid)
    @printf("  Cycle 24 — MAE=%.3f  RMSE=%.3f\n", val_mae, val_rmse)

    # ── One-step-ahead forecast: Cycle 25 (test) ─────────────────────────────
    println("[11] One-step-ahead forecast: Cycle 25 (test)...")
    full_series_tv = vcat(splits.train, splits.valid, splits.test)
    n_trainvalid   = length(splits.train) + length(splits.valid)
    forecast_test  = Float32[]
    Flux.reset!(model);  Flux.testmode!(model)
    for i in 1:length(splits.test)
        x = reshape(full_series_tv[n_trainvalid+i-window_size:n_trainvalid+i-1],
                    window_size, 1, 1) |> DEVICE
        ŷ = USE_GPU ? Array(model(x))[1] : model(x)[1]
        push!(forecast_test, ŷ)
    end
    test_mae, test_rmse = compute_metrics(splits.test, forecast_test)
    @printf("  Cycle 25 — MAE=%.3f  RMSE=%.3f\n", test_mae, test_rmse)
    @printf("  Baseline (Pala & Atici 2019) — RMSE=35.9\n")

    # ── Retrospective validation: Cycle 25 maximum ───────────────────────────
    # SILSO/STCE confirmed SSN = 160.9 in October 2024 as the Cycle 25 maximum.
    c25_max_date = Date(2024, 10, 1)
    c25_max_obs  = 160.9f0
    idx_max = findfirst(d -> d >= c25_max_date,
                        data.dates[splits.idx_c25:end])
    if !isnothing(idx_max)
        pred_at_max = forecast_test[idx_max]
        err_pct     = abs(pred_at_max - c25_max_obs) / c25_max_obs * 100
        @printf("  Cycle 25 max (Oct 2024): observed=%.1f | predicted=%.1f | error=%.1f%%\n",
                c25_max_obs, pred_at_max, err_pct)
    end

    # ── Deterministic Cycle 26 projection (mean trajectory) ─────────────────
    println("\n[12] Cycle 26 autoregressive projection ($future_steps months)...")
    full_series  = vcat(splits.train, splits.valid, splits.test)
    future_preds = forecast_future(model, full_series, window_size, future_steps)
    last_date    = data.dates[end]
    future_dates = [last_date + Month(i) for i in 1:future_steps]

    # ── MC Dropout post-hoc uncertainty for Cycle 26 ─────────────────────────
    # Strategy B: model was trained WITHOUT Dropout (preserving full accuracy).
    # Dropout is injected post-hoc only for stochastic sampling of Cycle 26.
    # This decouples evaluation quality from uncertainty quantification.
    # Reference: Gal & Ghahramani (2016), ICML.
    println("\n[13] MC Dropout post-hoc uncertainty (N=$mc_n_passes, p=0.10)...")
    mc_mean, mc_p05, mc_p95 = mc_dropout_posthoc(
        model, full_series, window_size, future_steps;
        p_drop=0.10f0, n_passes=mc_n_passes)

    # ── Save all figures ──────────────────────────────────────────────────────
    println("\n[14] Generating figures...")
    plot_forecast_cycles(data, splits, forecast_valid, forecast_test,
                         val_mae, val_rmse, test_mae, test_rmse)
    plot_full_series(data, splits, forecast_valid, forecast_test,
                     future_dates, mc_mean)          # MC mean as best estimate
    plot_cycle26_uncertainty(data, mc_mean, mc_p05, mc_p95, future_dates)

    println("\nFigures saved (PDF vector + PNG raster):")
    println("  smoothing | spectrum | acf | dtw | lr_finder |")
    println("  training_curve | forecast | full_series |")
    println("  cycle26_uncertainty  ← NEW: MC Dropout 90% CI")

    return model, data, splits, forecast_valid, forecast_test,
           future_preds, mc_mean, mc_p05, mc_p95
end

using Printf
main()
