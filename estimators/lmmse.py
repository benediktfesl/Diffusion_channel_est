import numpy as np
from modules.utils import toeplitz


def mp_eval(obj, y, toep, genie, A=None):
    if genie:
        hest = obj.estimate_genie(y, toep, A)
    else:
        hest = obj.estimate_global(y, toep, A)
    return hest #return_mse(hest, h_true)


class LMMSE:
    def __init__(self, snr):
        self.snr = snr
        self.rho = 10 ** (0.1 * snr)
        self.sigma2 = 1 / self.rho

    def estimate_genie(self, y, t, A=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        for b in range(n_batches):
            if type(t) is tuple:
                t_rx, t_tx = t
                C_rx = toeplitz(t_rx[b, :])
                C_tx = toeplitz(t_tx[b, :])
                C = np.kron(C_tx, C_rx)
            else:
                C = toeplitz(t[b, :]).T  # get full cov matrix
            CAh = C @ A.conj().T
            Cy = A @ CAh + 1 / self.rho * np.eye(A.shape[0])
            Cinv = np.linalg.pinv(Cy, hermitian=True)
            hest[b, :] = CAh @ Cinv @ y[b, :]
        return hest


    def estimate_global(self, y, C, A=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=complex)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        Cy = A @ C @ A.conj().T + 1 / self.rho * np.eye(A.shape[0])
        Cinv = np.linalg.pinv(Cy, hermitian=True)
        prod = C @ A.conj().T @ Cinv
        for b in range(n_batches):
            hest[b, :] = prod @ y[b, :]
        return hest
