from .base import HparamsBase


class HparamsMBLD(HparamsBase):
    def __init__(self, dataset, dataset_type, sampler):

        self.total_steps = 256
        self.sample_steps = 256
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.
        self.temp = 1.0
        self.weight_decay = 0.0
        self.beta = 0.1
        self.loss_final = 'weighted'
        self.beta_type = 'linear'
        self.epsilon = 0.0
        self.grad_norm = 0.0
        self.drop_path = 0.0
        self.p_flip = False
        self.focal = 0
        self.aux = 0.0 # better numerical statbility
        self.norm_first = False
        self.use_softmax = False
        self.use_tanh = False
        self.update_freq = 1

        self.load_model_step = -1
        self.load_model_dir = ''
        self.guidance = False
        self.omega = 0.0
        self.root_path = ''

        self.allow_mismatch = False
        self.cross = False
        self.use_gcc = False
        self.reset_step = False

        self.init_scale = 0
        self.optim_eps = 1e-8
        self.reset_scaler = False

        self.clip_version = 'ViT-B/32'

        self.max_length = 500
        
        super().__init__(dataset, dataset_type, sampler)
        
        if sampler == "trans":
            if self.dataset.startswith("motionx"):
                self.batch_size = 8
                self.bert_n_emb = 256
                self.bert_n_head = 4
                self.bert_n_layers = 12
                self.block_size = 13
                self.lr = 2e-4
                self.warmup_iters = 10000
                self.num_classes = 1000
    
            else:
                raise KeyError(f"Defaults not defined for Bernoulli diffusion model on dataset: {self.dataset}")
            
            
        elif sampler == "mdm":
            # self.legacy = False
            # self.modeltype = None

            self.data_rep = 'hml_vec'
            self.lr = 1e-5
            self.warmup_iters = 10000

            # self.pose_rep = 'rot6d'
            # self.glob = True
            # self.glob_rot = True
            # self.translation = True

            # self.num_actions = 1
            # self.ablation = None
            # self.action_emb

            if self.dataset_type == "newjoints":
                # self.nfeats = 128
                self.njoints = 13

                self.latent_dim = 512
                self.ff_size = 1024
                self.num_layers = 8
                self.num_heads = 4
                self.dropout = 0.1
                self.activation = 'gelu'
                self.clip_dim = 512
                self.normalize_encoder_output = False
                self.cond_mode = 'text'
                self.cond_mask_prob = 0.1
                self.arch = 'trans_enc'
                self.emb_trans_dec = False
            
            elif self.dataset_type == "smplx":
                self.njoints = 322
                self.nfeats = 1

                self.latent_dim = 512
                self.ff_size = 1024
                self.num_layers = 8
                self.num_heads = 4
                self.dropout = 0.1
                self.activation = 'gelu'
                self.clip_dim = 512
                self.normalize_encoder_output = False
                self.cond_mode = 'text'
                self.cond_mask_prob = 0.1
                self.arch = 'trans_enc'
                self.emb_trans_dec = False



# arguments for all sampler models
def add_mbld_args(parser):
    parser.add_argument("--ae_load_dir", type=str, required=True)
    parser.add_argument("--ae_load_step", type=int, required=True)
    parser.add_argument("--attn_pdrop", type=float)
    parser.add_argument("--bert_n_emb", type=int)
    parser.add_argument("--bert_n_head", type=int)
    parser.add_argument("--bert_n_layers", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--embd_pdrop", type=float)
    parser.add_argument("--resid_pdrop", type=float)
    parser.add_argument("--sample_block_size", type=int)
    parser.add_argument("--sampler", type=str, required=True, choices=["mdm", "trans"])
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--sample_steps", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--warmup_iters", type=int)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--drop_path", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--act", type=str)
    parser.add_argument("--cls_tkn", action="store_true")
    parser.add_argument("--all_steps", action="store_true")
    parser.add_argument("--cos", action="store_true")
    parser.add_argument("--reduce", type=int)
    parser.add_argument("--loss_final", type=str, choices=["mean", "weighted"])
    parser.add_argument("--beta_type", type=str)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--grad_norm", type=float)
    parser.add_argument("--p_flip", action="store_true")
    parser.add_argument("--focal", type=float)
    parser.add_argument("--aux", type=float)
    parser.add_argument("--use_softmax", action="store_true")
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--update_freq", type=int)
    parser.add_argument("--load_model_step", type=int)
    parser.add_argument("--load_model_dir", type=str)
    parser.add_argument("--omega", type=float)
    parser.add_argument("--allow_mismatch", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--use_gcc", action="store_true")
    parser.add_argument("--reset_step", action="store_true")
    parser.add_argument("--init_scale", type=float)
    parser.add_argument('--optim_eps', type=float)
    parser.add_argument("--reset_scaler", action="store_true")
    parser.add_argument("--conditioned", action="store_true")
