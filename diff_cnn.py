"""
Train and test script for the DMCE.
"""
from DMCE import utils, DiffusionModel, Trainer, Tester, CNN
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DMCE.utils import cmplx2real

CUDA_DEFAULT_ID = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cpu', type=str)

    # get the used device
    args = parser.parse_args()
    device = args.device

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_dim = 64 # RX antennas
    n_dim2 = 16 # TX antennas
    num_train_samples = 100_000
    num_val_samples = 10_000  # must not exceed size of training set
    num_test_samples = 10_000
    seed = 453451

    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = True # learn channel distribution in angular domain through Fourier transform

    # set data params
    ch_type = '3gpp' # {quadriga_LOS, 3gpp}
    n_path = 3
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'
    complex_data = True

    data_train, data_val, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                     n_antennas_tx=n_dim2, n_train_ch=num_train_samples, n_val_ch=num_val_samples,
                                     n_test_ch=num_test_samples, return_toep=False)
    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_train = np.reshape(data_train, (-1, n_dim, n_dim2), 'F')
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
        data_val = np.reshape(data_val, (-1, n_dim, n_dim2), 'F')
    data_train = torch.from_numpy(np.asarray(data_train[:, None, :]))
    data_train = cmplx2real(data_train, dim=1, new_dim=False).float()
    data_val = torch.from_numpy(np.asarray(data_val[:, None, :]))
    data_val = cmplx2real(data_val, dim=1, new_dim=False).float()
    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()
    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # set data params
    cwd = os.getcwd()
    bin_dir = path.join(cwd, 'bin')
    data_shape = tuple(data_train.shape[1:])

    # data parameter dictionary, which is saved in 'sim_params.json'
    data_dict = {
        'bin_dir': str(bin_dir),
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'num_test_samples': num_test_samples,
        'train_dataset': ch_type,
        'test_dataset': ch_type,
        'n_antennas': n_dim,
        'mode': mode,
        'data_shape': data_shape,
        'complex_data': complex_data
    }

    # set Diffusion model params
    num_timesteps = 100 #int(np.random.choice([100, 300, 500, 1_000, 2_000]))
    loss_type = 'l2'
    which_schedule = 'linear'

    max_snr_dB = 40
    beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    if num_timesteps == 5:
        beta_end = 0.95  # -22.5dB
    elif num_timesteps == 10:
        beta_end = 0.7  # -22.5dB
    elif num_timesteps == 50:
        beta_end = 0.2  # -22.5dB
    elif num_timesteps == 100:
        beta_end = 0.1 # -22.5dB
    elif num_timesteps == 300:
        beta_end = 0.035  # -23dB
    elif num_timesteps == 500:
        beta_end = 0.02 #-22dB
    elif num_timesteps == 1_000:
        beta_end = 0.01 #-22dB
    elif num_timesteps == 10_000:
        beta_end = 0.001 #-24dB
    else:
        beta_end = 0.035
    objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
    loss_weighting = False # bool(np.random.choice([True, False]))
    clipping = False
    reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
    reverse_add_random = False  # True: PDF Sampling method | False: Reverse Mean Forwarding method

    # diffusion model parameter dictionary, which is saved in 'sim_params.json'
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random
    }

    kernel_size = (3, 3)
    n_layers_pre = 2
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 2
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 3
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 2
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    # batch_norm = True
    cnn_dict = {
        'data_shape': data_shape,
        'n_layers_pre': n_layers_pre,
        'n_layers_post': n_layers_post,
        'ch_layers_pre': ch_layers_pre,
        'ch_layers_post': ch_layers_post,
        'n_layers_time': n_layers_time,
        'ch_init_time': ch_init_time,
        'kernel_size': kernel_size,
        'mode': mode,
        'batch_norm': batch_norm,
        'downsamp_fac': downsamp_fac,
        'device': device,
    }

    # set Trainer params
    batch_size = 128
    lr_init = 1e-4
    lr_step_multiplier = 1.0
    epochs_until_lr_step = 150
    num_epochs = 500
    val_every_n_batches = 2000
    num_min_epochs = 50
    num_epochs_no_improve = 20
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)

    # Trainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        'fft_pre': fft_pre,
    }

    # set Tester params
    batch_size_test = 512
    criteria = ['nmse']

    # Tester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': batch_size_test,
        'criteria': criteria,
        'complex_data': complex_data,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
    }

    # create result directory
    os.makedirs(dir_result, exist_ok=True)

    # instantiate CNN, DiffusionModel, Trainer and Tester
    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    tester = Tester(diffusion_model, data_test, **tester_dict)

    # Print number of trainable parameters
    print(f'Number of trainable model parameters: {diffusion_model.num_parameters}')

    # other parameters dictionary, which is saved in 'sim_params.json'
    misc_dict = {'num_parameters': diffusion_model.num_parameters}

    # save the simulation parameters as a JSON file
    sim_dict = {
        'data_dict': data_dict,
        'diff_model_dict': diff_model_dict,
        'unet_dict': cnn_dict,
        'trainer_dict': trainer_dict,
        'tester_dict': tester_dict,
        'misc_dict': misc_dict
    }

    utils.save_params(dir_result=dir_result, filename='sim_params', params=sim_dict)

    # run training routine
    train_dict = trainer.train()
    utils.save_params(dir_result=dir_result, filename='train_results', params=train_dict)

    params = dict()
    params['dim'] = n_dim
    params['dim2'] = n_dim2
    params['data_train'] = num_train_samples
    params['data_test'] = num_test_samples
    params['data_val'] = num_val_samples
    params['epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['lr_start'] = lr_init
    params['lr_step_mult'] = lr_step_multiplier
    params['epochs_until_lr_step'] = epochs_until_lr_step
    params['timesteps'] = num_timesteps
    params['beta_start'] = beta_start
    params['beta_end'] = beta_end
    params['snr_low'] = diffusion_model.snrs_db.cpu().detach().numpy()[-1]
    params['snr_high'] = diffusion_model.snrs_db.cpu().detach().numpy()[0]
    params['dataset_train'] = ch_type
    params['dataset_test'] = ch_type
    params['schedule'] = which_schedule
    params['kernel_size'] = kernel_size
    params['timestamp'] = timestamp
    params['trained_epochs'] = train_dict['trained_epochs']
    params['num_min_epochs'] = num_min_epochs
    params['num_epochs_no_improve'] = num_epochs_no_improve
    params['loss_weighting'] = loss_weighting
    params['n_layers_pre'] = n_layers_pre
    params['ch_layers_pre'] = ch_layers_pre
    params['n_layers_post'] = n_layers_post
    params['ch_layers_post'] = ch_layers_post
    params['n_layers_time'] = n_layers_time
    params['ch_init_time'] = ch_init_time
    params['num_learnable_params'] = diffusion_model.num_parameters
    params['fft_pre'] = fft_pre
    params['batch_norm'] = batch_norm
    params['downsamp_fac'] = downsamp_fac

    params['seed'] = seed
    os.makedirs('./results/dm_est/', exist_ok=True)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_params.csv'
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
           writer.writerow([key, value])


    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_loss.png'
    plt.figure()
    plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss')
    plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss')
    #plt.plot(range(1, params['epochs'] + 1), losses_all_test, label='val-loss')
    plt.legend(['train-loss', 'val-loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name)

    # run testing routine
    test_dict = tester.test()

    if return_all_timesteps:
        # plot all curves
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                    f'T={num_timesteps}_perstep.png'
        plt.figure()
        lines = []
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            mse_list_allsteps = test_dict[criteria[0]]['NMSEs_total_power'][isnr]
            snr_now = test_dict[criteria[0]]['SNRs'][isnr]
            n_timesteps_eval = len(mse_list_allsteps)
            lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps, label=f'SNR = {int(snr_now)}')
            #plt.legend([f'SNR = {int(snr_now)}'])
            plt.xlabel('Timesteps')
            plt.ylabel('nMSE')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.savefig(file_name)

        # save all mses
        mse_list = list()
        mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
        mse_list[-1].insert(0, 'SNR')
        mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
        mse_list[-1].insert(0, 'nmse_dm')
        mse_list = [list(i) for i in zip(*mse_list)]
        print(mse_list)
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}_perstep.csv'
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(mse_list)

        # remove all mses except last to save it later
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            test_dict[criteria[0]]['NMSEs_total_power'][isnr] = test_dict[criteria[0]]['NMSEs_total_power'][isnr][-1]

    mse_list = list()
    mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    utils.save_params(dir_result=dir_result, filename='test_results', params=test_dict)


if __name__ == '__main__':
    main()
