import torch 
import torch.nn as nn

class NoQuantizer(nn.Module):
    def forward(self, h, deterministic):
        return h, torch.tensor(0.0).detach(), {"binary_code": h.detach()}, h.detach()


class BinaryQuantizer(nn.Module):
    def forward(self, h, deterministic):
        sigma_h = torch.sigmoid(h)
        binary = torch.bernoulli(sigma_h)
        aux_binary = binary.detach() + sigma_h - sigma_h.detach()
        #return aux_binary, torch.tensor(0.0, device=self.device).detach()
        return aux_binary, torch.tensor(0.0).detach(), {"binary_code": binary.detach()}, binary.detach()


class BinaryVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv1d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh


    def embed_code(self, code_b):
        z_q = torch.einsum("b n h w, n d -> b d h w", code_b, self.embed.weight)
        return z_q


    def quantizer(self, x, deterministic=False):
        # assert every value in x is between 0 and 1
        assert (x >= 0.0).all() and (x <= 1.0).all(), f"Values in x are not between 0 and 1: {x.min()}, {x.max()}"

        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)


    def forward(self, h, deterministic=False):
        # print('QUANTIZER FORWARD')
        # print("quantizer input shape: ", h.shape) # torch.Size([bs, 4, 13])
        # print("projection module", self.proj) # conv1d(4, 32, 1, 1) + sigmoid
        z = self.proj(h)
        # print('self.codebook_size: ', self.codebook_size) # 32
        # print('self.emb_dim: ', self.emb_dim) # 4
        # print("z shape: ", z.shape) # torch.Size([bs, 32, 13])

        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        # print("compute codebook loss")
        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()
        # print("loss: ", code_book_loss) # tensor(0.2499)

        # print("Apply bernoulli sampling")
        z_b = self.quantizer(z, deterministic=deterministic)
        z_flow = z_b.detach() + z - z.detach()
        # print("z_b shape: ", z_b.shape) # torch.Size([bs, 32, 13])
        # print("z_flow shape: ", z_flow.shape) # torch.Size([bs, 32, 13])
        # print("z_b: ", z_b) # binary tensor
        # print("z_flow: ", z_flow) # binary-float tensor with gradients

        # print("Apply code embedding")
        z_q = torch.einsum("b n w, n d -> b d w", z_flow, self.embed.weight)
        # print("embedding weights shape", self.embed.weight.shape) # torch.Size([32, 4])

        # return z_q,  code_book_loss, {
        #     "binary_code": z_b.detach()
        # }, z_b.detach()

        # print("z_q shape: ", z_q.shape) # torch.Size([bs, 4, 13])
        # print("z_q: ", z_q) # embedded binary-float tensor with gradients

        return z_q,  code_book_loss, {"binary_code": z_b.detach()}, z_b.detach()