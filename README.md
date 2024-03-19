# Diffusion-based Channel Estimation 
Source code of the paper 
>B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,
>"Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation," 2024, arXiv preprint: 2403.03545.
<br>
Link to the paper: https://arxiv.org/abs/2403.03545

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
