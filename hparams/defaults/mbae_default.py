from .base import HparamsBase


class HparamsMBAE(HparamsBase):
    def __init__(self, dataset, dataset_type):
        super().__init__(dataset, dataset_type)
        # defaults that are same for all datasets
        self.base_lr = 4.5e-6
        self.beta = 0.25
        self.path_to_data = None
        self.code_weight = 0.0
        self.norm_first = False
        self.use_tanh = False
        self.deterministic = False
        self.gen_mul = 1.0


        if self.dataset_type == 'newjoints':
            self.attn_resolutions = [13,26]
            self.batch_size = 4
            self.ch_mult = [1, 2, 4]
            self.codebook_size = 32
            self.emb_dim = 4
            self.resolution = 52
            self.latent_shape = [1, 4, 13]
            self.n_channels = 3
            self.nf = 128
            self.code_weight = 1.0
            self.mse_weight = 1.0
            self.l1_weight = 1.0
            self.res_blocks = 2
            self.key='motion'

        elif self.dataset_type == 'joints':
            self.attn_resolutions = [13,26]
            self.batch_size = 4
            self.ch_mult = [1, 2, 4]
            self.codebook_size = 128
            self.emb_dim = 16
            self.resolution = 52
            self.latent_shape = [1, 16, 13]
            self.n_channels = 3
            self.nf = 128
            self.code_weight = 0.8
            self.mse_weight = 1.0
            self.l1_weight = 0.1
            self.res_blocks = 2
            self.key='motion'

        elif self.dataset_type == 'smplx':
            self.attn_resolutions = [40,80]
            self.batch_size = 4
            self.ch_mult = [1, 1, 2, 4]
            self.codebook_size = 256
            self.emb_dim = 16
            self.resolution = 322
            self.latent_shape = [1, 16, 40]
            self.n_channels = 1
            self.nf = 128
            self.code_weight = 1.0
            self.mse_weight = 1.0
            self.l1_weight = 1.0
            self.res_blocks = 2
            self.key='motion'

        elif self.dataset_type == 'newjointvecs':
            self.attn_resolutions = [16,32]
            self.batch_size = 4
            self.ch_mult = [1, 1, 1, 2, 4]
            self.codebook_size = 64
            self.emb_dim = 16
            self.resolution = 263
            self.latent_shape = [1, 16, 16]
            self.n_channels = 1
            self.nf = 64
            self.code_weight = 1.0
            self.mse_weight = 1.0
            self.l1_weight = 1.0
            self.res_blocks = 2
            self.key='motion'

        else:
            raise KeyError(f'Defaults not defined for BinaryAE model on dataset: {self.dataset}')


def add_mbae_args(parser):
    parser.add_argument('--attn_resolutions', nargs='+', type=int)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--codebook_size', type=int)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--horizontal_flip', const=True, action='store_const', default=False)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--latent_shape', nargs='+', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--nf', type=int)
    parser.add_argument('--res_blocks', type=int)
    parser.add_argument('--code_weight', type=float)
    parser.add_argument('--mse_weight', type=float)
    parser.add_argument('--l1_weight', type=float)
    parser.add_argument('--gen_mul', type=float)
    parser.add_argument('--norm_first', action="store_true")
    parser.add_argument('--use_tanh', action="store_true")
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--cp_data', action="store_true")
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--key', type=str)