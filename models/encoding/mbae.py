import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoding.modules.encoder import Encoder, Generator
from models.encoding.modules.quantizer import BinaryVectorQuantizer
from models.encoding.modules.losses import MSELoss, L1Loss

class MotionBinaryAutoEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.codebook_size = H.codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.resolution = H.resolution
        self.attn_resolutions = H.attn_resolutions
        self.quantizer_type = H.quantizer
        self.beta = H.beta
        self.gumbel_num_hiddens = H.emb_dim
        self.deterministic = H.deterministic
        self.code_w = H.code_weight
        self.mse_w = H.mse_weight
        self.l1_w = H.l1_weight
        self.key = H.key
        self.dataset_type = H.dataset_type

        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        self.quantize = BinaryVectorQuantizer(self.codebook_size, self.embed_dim, self.embed_dim, use_tanh=H.use_tanh)
        self.generator = Generator(H)
        self.mseloss = MSELoss()
        self.l1loss = L1Loss()

        if self.dataset_type == 'smplx':
            self.linear_input = nn.Linear(322, 320)
            self.linear_output = nn.Linear(320, 322)

        print(self.encoder)
        print(self.quantize)
        print(self.generator)


    def get_input(self, batch, lengths, k=None):
        """
        Get input from batch as (B, F, J, C) tensor and permutes it to (B*F, C, J).
        Where B is bach, F is frames, J is joints and C is channels or coordinates.
        """
        # print("Keys: ", batch.keys())
        # print(type(batch))
        # print(type(batch[k]))
        # print(k)
        #print(batch[k].shape)
        # print("batch lengths: ", batch['length'])
        # print("GET INPUT")
        if isinstance(batch, torch.Tensor):
            x = batch
        else:
            x = batch[k]
        # print("x len: ", len(x))

        elements = []
        # print("x shape: ", x.shape)
        if self.dataset_type == 'newjoints':
            for idx in range(x.shape[0]):
                element = x[idx, :, :, :lengths[idx]]
                # print("element shape: ", element.shape)
                # try:
                element = element.permute(2,0,1)
                # except:
                #     assert len(element.shape) == 2
                #     # Add a dummy dimension for channels
                #     element = element.unsqueeze(0)
                #     element = element.permute(2,0,1)

                elements.append(element)
                # print("element shape: ", element.shape)
            x = torch.cat(elements, dim=0)
            # print("x shape: ", x.shape)
            return x.float()

        elif self.dataset_type == 'smplx':
            for idx in range(x.shape[0]):
                element = x[idx, :lengths[idx], :]
                # print("element shape: ", element.shape)
                try:
                    assert len(element.shape) == 2
                    # Add a dummy dimension for channels
                    element = element.unsqueeze(0)
                    element = element.permute(1,0,2)
                except:
                    element = element.permute(1,0,2)

                elements.append(element)
                # print("element shape: ", element.shape)
            x = torch.cat(elements, dim=0)
            # print("x shape: ", x.shape)
            return x.float()


    def forward(self, x, lengths=None, code_only=False, code=None, device=None):
        # print('x shape: ', x.shape)

        # assert shape is (n, 3, 52)
        # assert len(x.shape) == 3 and x.shape[1] == 3 and x.shape[2] == 52, f"Shape of input is not (n, 3, 52): {x.shape}"
        # assert there are no NaNs
        # assert (x != x).sum() == 0, f"Input contains NaNs: {batch}"
        if code is None:
            assert lengths is not None, "Lengths must be provided"
            # print('AUTOENCODER')
            # print('x shape: ', x.keys())
            x_input = self.get_input(x, lengths, self.key)
            # print('x_input shape: ', x_input.shape)
            if device is not None:
                x_input = x_input.to(device)
            if self.dataset_type == 'smplx':
                x = self.linear_input(x_input)
                # print('x shape: ', x.shape)
            x = self.encoder(x_input)
            # print('x shape: ', x.shape)
            quant, codebook_loss, quant_stats, binary = self.quantize(x, deterministic=self.deterministic)
            if code_only:
                return binary
        else:
            quant = torch.einsum("b n w, n d -> b d w", code, self.quantize.embed.weight)
            codebook_loss, quant_stats = None, None
            x_input = None
        x = self.generator(quant)
        # print('x shape: ', x.shape)

        if self.dataset_type == 'smplx':
            x = self.linear_output(x)

        return x, codebook_loss, quant_stats, x_input


    def training_step(self, batch, device='cpu', return_input=False):
        stats = {}

        x_hat, codebook_loss, quant_stats, x_input = self(batch, device=device)
        # print('x_hat shape: ', x_hat.shape)
        # print('x_input shape: ', x_input.shape)

        # get reconstruction loss
        if self.l1_w > 0.0:
            l1_loss = self.l1loss(x_input, x_hat)
            l1_loss = torch.mean(l1_loss)
        else:
            l1_loss = torch.tensor(0.0).to(x_input.device).detach()

        if self.mse_w > 0.0:
            mse_loss = self.mseloss(x_input, x_hat)
            mse_loss = torch.mean(mse_loss)
        else:
            mse_loss = torch.tensor(0.0).to(x_input.device).detach()

        # get loss
        loss = self.mse_w * mse_loss + self.l1_w * l1_loss + self.code_w * codebook_loss

        stats["loss"] = loss
        stats["l1_loss"] = l1_loss.item()
        stats["mse_loss"] = mse_loss.item()
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]


        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"].item()

        if return_input:
            return x_hat, stats, x_input
        return x_hat, stats


    @torch.no_grad()
    def validation_step(self, batch, step, device='cpu'):
        stats = {}

        x = self.get_input(batch, self.key).to(device)

        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.quantizer_type == "gumbel":
            self.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.quantize.temperature

        x_hat, codebook_loss, quant_stats = self(x)

        # get reconstruction loss
        if self.l1_w > 0.0:
            l1_loss = self.L1Loss(x, x_hat)
            l1_loss = torch.mean(l1_loss)
        else:
            l1_loss = torch.tensor(0.0).to(x.device).detach()

        if self.mse_w > 0.0:
            mse_loss = self.MSELoss(x, x_hat)
            mse_loss = torch.mean(mse_loss)
        else:
            mse_loss = torch.tensor(0.0).to(x.device).detach()

        # get loss
        loss = self.mse_w * mse_loss + self.l1_w * l1_loss + self.code_w * codebook_loss

        stats["l1_loss"] = l1_loss.item()
        stats["mse_loss"] = mse_loss.item()
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]

        return x_hat, stats


    @torch.no_grad()
    def log_motions(self):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement log motions")


    @torch.no_grad()
    def smplx2joint(self, smplx):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement smplx2joint")


    @torch.no_grad()
    def joint2newjoint(self, joint):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement joint2newjoint")


