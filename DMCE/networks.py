import torch
from torch import nn
import math
from typing import Union
from DMCE import utils


def get_positional_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Creates the DM time embedding from an integer time step

    Parameters
    ----------
    t : Tensor of shape [batch_size]
        timesteps of the corresponding data samples
    dim : int
        dimension of the resulting embedding
    Returns
    -------
    t_emb : Tensor of shape [batch_size, dim]
        time embeddings for each data sample
    """

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(- emb * torch.arange(half_dim, device=t.device))
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

    # if dim is an odd number, pad the last entry of the embedding vector with zeros
    if dim % 2 != 0:
        emb = torch.nn.functional.pad(emb, (0, 1), 'constant', 0)
    return emb



class CNN(nn.Module):
    def __init__(self,
                 data_shape: tuple,
                 n_layers_pre: int = 1,
                 n_layers_post: int = 1,
                 ch_layers_pre: tuple = (2, 2),
                 ch_layers_post: tuple = (2, 2),
                 n_layers_time: int = 1,
                 ch_init_time: int = 16,
                 kernel_size: tuple = (3, ),
                 mode: str = '1D',
                 batch_norm: bool = False,
                 downsamp_fac: int = 1,
                 stride: int = 1,
                 padding_mode: str = 'zeros',
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__()
        self.data_shape = data_shape
        self.n_layers_pre = n_layers_pre
        self.n_layers_post = n_layers_post
        self.ch_layers_pre = ch_layers_pre
        self.ch_layers_post = ch_layers_post
        self.n_layers_time = n_layers_time
        self.ch_init_time = ch_init_time
        self.kernel_size = kernel_size
        self.mode = mode
        self.batch_norm = batch_norm
        self.downsamp_fac = downsamp_fac
        self.stride = stride
        self.padding_mode = padding_mode
        self.device = utils.set_device(device)

        #self.dim_time = np.prod(data_shape[1:])
        self.dim_time = ch_layers_pre[-1]
        ch_time = None
        if n_layers_time == 0:
            ch_time = (2*self.dim_time, )
        elif n_layers_time == 1:
            ch_time = (ch_init_time, 2*self.dim_time)
        elif n_layers_time == 2:
            ch_time = (ch_init_time, self.dim_time, 2*self.dim_time)
        elif n_layers_time == 3:
            ch_time = (ch_init_time, self.dim_time, self.dim_time, 2 * self.dim_time)
        else:
            raise NotImplementedError

        # Time embedding related functionalities, computing the base time embedding
        self.time_embedding_func = lambda t: get_positional_embedding(t, ch_time[0])
        self.time_mlp = nn.Sequential().to(device=device)
        for i in range(n_layers_time):
            self.time_mlp.add_module(f'time_linear{i+1}', nn.Linear(ch_time[i], ch_time[i+1], device=self.device))
            if i < n_layers_time - 1:
                self.time_mlp.add_module(f'act_time{i+1}', nn.ReLU())


        self.cnn_pre = nn.Sequential().to(device=device)
        for i in range(n_layers_pre):
            if mode == '1D':
                self.cnn_pre.add_module(f'conv_pre{i}', nn.Conv1d(ch_layers_pre[i], ch_layers_pre[i+1], stride=stride,
                                   kernel_size=kernel_size, padding='same',device=device))
            else:
                self.cnn_pre.add_module(f'conv_pre{i}', nn.Conv2d(ch_layers_pre[i], ch_layers_pre[i+1], stride=stride,
                                   kernel_size=kernel_size, padding='same', device=device))
            if i < n_layers_pre - 1:
                if batch_norm and mode == '2D':
                    self.cnn_pre.add_module(f'batchnorm_pre{i+1}', nn.BatchNorm2d(num_features=ch_layers_pre[i+1], device=device))
                self.cnn_pre.add_module(f'act_pre{i+1}', nn.ReLU())

        self.cnn_post = nn.Sequential().to(device=device)
        for i in range(n_layers_post):
            if mode == '1D':
                self.cnn_post.add_module(f'conv_post{i}', nn.Conv1d(ch_layers_post[i], ch_layers_post[i + 1],
                                                         stride=stride,kernel_size=kernel_size, padding='same',
                                                         device=device))
            else:
                self.cnn_post.add_module(f'conv_post{i}', nn.Conv2d(ch_layers_post[i], ch_layers_post[i + 1],
                                                        stride=stride, kernel_size=kernel_size, padding='same',
                                                        device=device))
            if i < n_layers_post - 1:
                if batch_norm and mode == '2D':
                    self.cnn_post.add_module(f'batchnorm_post{i+1}', nn.BatchNorm2d(num_features=ch_layers_post[i+1], device=device))
                self.cnn_post.add_module(f'act_post{i+1}', nn.ReLU())


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # compute the time embedding for all timesteps
        t_emb = self.time_mlp(self.time_embedding_func(t))
        scale = t_emb[:, :self.dim_time]
        shift = t_emb[:, self.dim_time:]

        x = self.cnn_pre(x)
        if self.mode == '1D':
            x = x + scale[:, :, None] * x + shift[:, :, None]
        else:
            x = x + scale[:, :, None, None] * x + shift[:, :, None, None]
        x = self.cnn_post(x)
        return x

