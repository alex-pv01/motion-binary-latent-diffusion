import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from model.encoding.modules.encoder import Encoder, Decoder
from model.encoding.modules.quantizer import BinaryQuantizer, BinaryVectorQuantizer
from utils.utils import instantiate_from_config

class MBVAEModel(nn.Module):
    """
    Main model class for Motion Binary Variational Autoencoder.
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 codebook_size=None,
                 emb_dim=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 key="motion",
                 colorize_nlabels=None,
                 monitor=None,
                 quantize="binary",
                 scheduler_config=None,
                 base_learning_rate=1e-6,
                 ):
        
        super().__init__()
        self.key = key
        #TODO: check if the encoder and decoder are correct and consider other architeectures as in mld
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quant_type = quantize
        #TODO: merge vector and binary quantization into one class
        if quantize == "vector":
            assert codebook_size is not None and emb_dim is not None, "Need to specify codebook_size and emb_dim for vector quantization."
            self.quantize = BinaryVectorQuantizer(codebook_size=codebook_size, emb_dim=emb_dim, num_hiddens=ddconfig["z_channels"])
        elif quantize == "binary":
            self.quantize = BinaryQuantizer()
        self.quant_conv = torch.nn.Conv1d(ddconfig["z_channels"], emb_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(emb_dim, ddconfig["z_channels"], 1)
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.base_learning_rate = base_learning_rate
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config



    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def encode(self, x):
        h = self.encoder(x)
        print("h shape: ", h.shape)
        h = self.quant_conv(h)
        print("h shape: ", h.shape)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        print("quant shape: ", quant.shape)
        dec = self.decoder(quant)
        return dec
    

    def forward(self, input):
        print("MBVAEModel forward")
        print("input shape: ", input.shape)
        quant, diff, _ = self.encode(input)
        print("quant shape: ", quant.shape)
        dec = self.decode(quant)
        print("dec shape: ", dec.shape)
        print("MBVAEModel forward end")
        return dec, diff


    def get_input(self, batch, k):
        """
        Get input from batch as (B, F, J, C) tensor and permutes it to (B*F, C, J).
        Where B is bach, F is frames, J is joints and C is channels or coordinates.
        """
        #print("Keys: ", batch.keys())
        #print(type(batch))
        #print(type(batch[k]))
        #print(k)
        #print(batch[k].shape)
        x = batch[k]
        elements = []
        for element in x:
            element = element.permute(0,2,1)
            elements.append(element)
            #print("element shape: ", element.shape)
        x = torch.cat(elements, dim=0)
        #print("x shape: ", x.shape)
        return x.float()
    

    def training_step(self, batch, batch_idx, device=None):
        x = self.get_input(batch, self.key).to(device)

        x_hat, diff = self(x)
        loss = self.loss(x_hat, x, diff)
        if self.use_scheduler:
            self.scheduler.step()
        return loss
    

    def validation_step(self, batch, batch_idx, device=None):
        x = self.get_input(batch, self.key).to(device)
        x_hat, diff = self(x)
        loss = self.loss(x_hat, x, diff)
        return loss
    

    def configure_optimizers(self):
        lr = self.base_learning_rate
        optim = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        if self.use_scheduler:
            raise NotImplementedError("TODO: implement scheduler")
            self.scheduler = LambdaLR(optim, **self.scheduler_config)
            return [optim], []
        else:
            return optim
        
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

    def log_motions(self):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement log motions")


    def smplx2joint(self, smplx):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement smplx2joint")
    

    def joint2newjoint(self, joint):
        #TODO: use scripts from tomato
        raise NotImplementedError("TODO: implement joint2newjoint")