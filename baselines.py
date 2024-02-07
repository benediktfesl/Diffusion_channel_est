import modules.utils as ut
import csv
import datetime
from estimators.lmmse import LMMSE, mp_eval
import numpy as np
import multiprocessing as mp
import os


def mp_gmm(obj, *args):
    return obj.estimate_from_y(*args)

def mp_omp(obj, *args):
    return obj.estimate(*args)


def main():
    n_processes = int(mp.cpu_count() / 2)
    print('Uses ' + str(n_processes) + ' processes')
    # prepare multiprocessing
    pool = mp.Pool(processes=n_processes)

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_antennas_rx = 64
    n_antennas_tx = 16
    n_train_ch = 100_000
    n_val_ch = 10_000  # must not exceed size of training set
    n_test_ch = 10_000
    snrs = [-15, -10, -5, 0, 5, 10, 15, 20]
    ch_type = 'quadriga_LOS'
    n_path = 3

    eval_LS = True
    eval_lmmse_glob = True
    eval_lmmse_genie = True

    channels_train, toep_train, channels_val, _, channels_test, toep_test = ut.load_or_create_data(ch_type=ch_type,
                            n_path=n_path, n_antennas_rx=n_antennas_rx, n_antennas_tx=n_antennas_tx,
                            n_train_ch=n_train_ch, n_val_ch=n_val_ch, n_test_ch=n_test_ch, return_toep=True)

    # vectorize channels
    n_antennas = n_antennas_rx * n_antennas_tx
    if ch_type.startswith('quadriga'):
        channels_train = np.reshape(channels_train, (-1, n_antennas), 'F')
        channels_test = np.reshape(channels_test, (-1, n_antennas), 'F')

    mse_list = list()
    mse_list.append(snrs.copy())
    mse_list[-1].insert(0, 'SNR')

    if eval_lmmse_glob:
        mse_list.append(['lmmse_glob'])
        cov = np.zeros([n_antennas, n_antennas], dtype=complex)
        for i in range(channels_train.shape[0]):
            cov = cov + np.expand_dims(channels_train[i, :], 1) @ np.expand_dims(channels_train[i, :].conj(), 0)
        cov = cov / channels_train.shape[0]
        eval_list_glob = list()
        for snr in snrs:
            y = ut.get_observation(channels_test, snr)
            eval_list_glob.append([LMMSE(snr), y, cov, False])
        res_glob_lmmse = pool.starmap(mp_eval, eval_list_glob)
        for it, res in enumerate(res_glob_lmmse):
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)


    if eval_LS:
        mse_list.append(['LS'])
        for snr in snrs:
            y = ut.get_observation(channels_test, snr)
            mse_act = np.sum(np.abs(y - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)


    if ch_type == '3gpp' and eval_lmmse_genie:
        mse_list.append(['lmmse_genie'])
        eval_list_genie = list()
        for snr in snrs:
            y = ut.get_observation(channels_test, snr)
            eval_list_genie.append([LMMSE(snr), y, toep_test, True])
        res_genie_lmmse = pool.starmap(mp_eval, eval_list_genie)
        for it, res in enumerate(res_genie_lmmse):
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)


    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    os.makedirs('./results/baselines/', exist_ok=True)
    file_name = f'./results/baselines/{date_time}_{ch_type}_path={n_path}_ant={n_antennas_rx}x{n_antennas_tx}_' \
                f'testdata={channels_test.shape[0]}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)


if __name__ == '__main__':
    main()

