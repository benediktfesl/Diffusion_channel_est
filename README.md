# Diffusion-Based Channel Estimation 
Source code of the paper 
>B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,
>"Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation," in IEEE Wireless Communications Letters, 2024.
<br>
Link to the paper: https://ieeexplore.ieee.org/document/10705115

## Abstract
This letter proposes a novel channel estimator based on diffusion models (DMs), one of the currently top-rated generative models, with provable convergence to the mean square error (MSE)-optimal estimator. A lightweight convolutional neural network (CNN) with positional embedding of the signal-to-noise ratio (SNR) information is designed to learn the channel distribution in the sparse angular domain. Combined
with an estimation strategy that avoids stochastic resampling and truncates reverse diffusion steps that account for lower SNR than the given pilot observation, the resulting DM estimator unifies low complexity and memory overhead. Numerical results exhibit
better performance than state-of-the-art estimators.

## Requirements
The code is tested with `Python 3.10` and `Pytorch 2.1.1`. For further details, see `environment.yml`.

## Instructions
1. Load channel data from https://syncandshare.lrz.de/getlink/fi93y1AnwmsvHrAGNqq5zX/ (password: Diffusion2024) and move it into folder `bin`.

2. To evaluate the pre-trained models used for the plots in the paper, run 
```
python load_and_eval_dm.py -d cuda:0
```

3. To train a DM from scatch and evaluate the performance afterward, run
```
python diff_cnn.py -d cuda:0
```

4. To evaluate the baseline estimators, run
```
python baselines.py
```

The code is based on the implementation of https://github.com/benediktfesl/Diffusion_MSE.
