"""
Train and test script for the DMCE.
"""
import os
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import DMCE
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

    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = True # learn channel distribution in angular domain through Fourier transform
    reverse_add_random = False # re-sampling in the reverse process

    # set data params
    ch_type = '3gpp' # {quadriga_LOS, 3gpp}
    n_path = 3
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'

    _, _, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                                             n_antennas_tx=n_dim2, n_train_ch=num_train_samples,
                                                             n_val_ch=num_val_samples,
                                                             n_test_ch=num_test_samples, return_toep=False)
    del _
    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()
    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # load the model parameter dictionaries
    cwd = os.getcwd()
    #which_dataset = dataset
    model_dir = os.path.join(cwd, './results/best_models_dm_paper', ch_type)
    sim_params = DMCE.utils.load_params(os.path.join(model_dir, 'sim_params'))
    cnn_dict = sim_params['unet_dict']
    diff_model_dict = sim_params['diff_model_dict']

    # manually set the correct device for this simulation
    cnn_dict['device'] = device

    # instantiate the neural network
    cnn = DMCE.CNN(**cnn_dict)

    # instantiate the diffusion model and give it a reference to the unet model
    diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)

    # load the parameters of the pre-trained model into the DiffusionModel instance
    model_path = os.path.join(model_dir, 'train_models')
    model_list = os.listdir(model_path)
    model_path = os.path.join(model_path, model_list[-1])
    model_params = torch.load(model_path, map_location=device)

    diffusion_model.load_state_dict(model_params['model'])

    # Tester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': 512,
        'criteria': ['nmse'],
        'complex_data': False,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
    }

    # instantiate the Tester and give it a reference to the diffusion model as well as testing data
    tester = DMCE.Tester(diffusion_model, data=data_test, **tester_dict)

    num_timesteps = sim_params['diff_model_dict']['num_timesteps']

    diffusion_model.reverse_add_random = reverse_add_random

    # call the test() function. This returns a dictionary with the testing stats.
    # Depending on the size of the test set, this might take a while.
    test_dict = tester.test()

    os.makedirs('./results/dm_est/', exist_ok=True)
    if return_all_timesteps:
        # plot all curves
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                    f'T={num_timesteps}_perstep_best.png'
        plt.figure()
        lines = []
        for isnr in range(len(test_dict['nmse']['NMSEs_total_power'])):
            mse_list_allsteps = test_dict['nmse']['NMSEs_total_power'][isnr]
            snr_now = test_dict['nmse']['SNRs'][isnr]
            n_timesteps_eval = len(mse_list_allsteps)
            lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps,
                                  label=f'SNR = {int(snr_now)}')
            #plt.legend([f'SNR = {int(snr_now)}'])
            plt.xlabel('Timesteps')
            plt.ylabel('nMSE')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.savefig(file_name)

        # save all mses
        list_snrs_all = test_dict['nmse']['SNRs'].copy()
        list_mses_all = test_dict['nmse']['NMSEs_total_power'].copy()
        for i in range(len(list_snrs_all)):
            n_timesteps_eval = len(list_mses_all[i])
            mse_list = list()
            mse_list.append(list(range(n_timesteps_eval))[::-1])
            mse_list[-1].insert(0, 't')
            mse_list.append(list_mses_all[i])
            mse_list[-1].insert(0, 'nmse_dm')
            mse_list = [list(i) for i in zip(*mse_list)]
            file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                        f'T={num_timesteps}_best_SNR={list_snrs_all[i]}.csv'
            with open(file_name, 'w') as myfile:
                wr = csv.writer(myfile, lineterminator='\n')
                wr.writerows(mse_list)

        # remove all mses except last to save it later
        for isnr in range(len(test_dict['nmse']['NMSEs_total_power'])):
            test_dict['nmse']['NMSEs_total_power'][isnr] = test_dict['nmse']['NMSEs_total_power'][isnr][-1]

    mse_list = list()
    mse_list.append(test_dict['nmse']['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict['nmse']['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_resamp={reverse_add_random}_best.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)


if __name__ == '__main__':
    main()