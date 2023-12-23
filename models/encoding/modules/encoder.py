import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []

        # Initial convolution
        blocks.append(nn.Conv1d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # Residual and downsampling blocks, with attention on smaller resolutions
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # Non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # Normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))
        blocks.append(nn.Conv1d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # print("Encoder input shape: ", x.shape)
        for i, block in enumerate(self.blocks):
            # print(f"Encoder block {i+1}: ", block)
            x = block(x)
            # 1st block: x = nn.Conv1d(in_channels=3, nf=128, kernel_size=3, stride=1, padding=1)
            # 1st output shape: torch.Size([b*l, 128, 52])
            # 2n and 3rd block: x = ResBlock(block_in_ch=128, block_out_ch=128) with conv1d of 128 channels and kernel size 3
            # 2n and 3rd output shape: torch.Size([b*l, 128, 52])
            # 4th block: x = Downsample(block_in_ch=128) with conv1d of 128 channels, kernel size 3 and stride 2
            # 4th output shape: torch.Size([b*l, 128, 26])
            # 5th block: x = ResBlock(block_in_ch=128, block_out_ch=256) with conv1d of 256 channels and kernel size 3
            # 5th output shape: torch.Size([b*l, 256, 26])
            # 6th and 8th block: x = AttnBlock(block_in_ch=256) with conv1d of 256 channels and kernel size 1
            # 6th and 8th output shape: torch.Size([b*l, 256, 26])
            # 7th block: x = ResBlock(block_in_ch=256, block_out_ch=256) with conv1d of 256 channels and kernel size 3
            # 7th output shape: torch.Size([b*l, 256, 26])
            # 9th block: x = Downsample(block_in_ch=256) with conv1d of 256 channels, kernel size 3 and stride 2
            # 9th output shape: torch.Size([b*l, 256, 13])
            # 10th block: x = ResBlock(block_in_ch=256, block_out_ch=512) with conv1d of 512 channels and kernel size 3
            # 10th output shape: torch.Size([b*l, 512, 13])
            # 11th, 13th and 15th block: x = AttnBlock(block_in_ch=512) with conv1d of 512 channels and kernel size 1
            # 11th, 13th and 15th output shape: torch.Size([b*l, 512, 13])
            # 12th, 14th and 16th block: x = ResBlock(block_in_ch=512, block_out_ch=512) with conv1d of 512 channels and kernel size 3
            # 12th, 14th and 16th output shape: torch.Size([b*l, 512, 13])
            # 17th block: x = GroupNorm(num_groups=4, num_channels=512, eps=1e-6, affine=True)
            # 17th output shape: torch.Size([b*l, 512, 13])
            # 18th block: x = nn.Conv1d(block_in_ch=512, out_channels=4, kernel_size=3, stride=1, padding=1)
            # 18th output shape: torch.Size([b*l, 4, 13])
            # print("Block output shape: ", x.shape)
        return x



class Generator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(H.res_blocks * H.gen_mul)
        self.resolution = H.resolution
        self.attn_resolutions = H.attn_resolutions
        self.in_channels = H.emb_dim
        self.out_channels = H.n_channels
        self.norm_first = H.norm_first
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        

        blocks = []
        # Initial conv
        if self.norm_first:
            blocks.append(Normalize(self.in_channels))
        blocks.append(nn.Conv1d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # Non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(Normalize(block_in_ch))
        blocks.append(nn.Conv1d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # used for calculating ELBO - fine tuned after training
        if H.gpu is not None:
            self.logsigma = nn.Sequential(
                                nn.Conv1d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv1d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
                            ).cuda()
        else:
            self.logsigma = nn.Sequential(
                                nn.Conv1d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv1d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
                            )

    def forward(self, x):
        # print("GENERATOR FORWARD")
        # print("Generator input shape: ", x.shape) # torch.Size([b*l, 4, 13])
        for i, block in enumerate(self.blocks):
            # print(f"Generator block {i}: ", block)
            x = block(x)
            # print("Block output shape: ", x.shape)
            # Block 0: GroupNorm 4, 4
            # Block 1: nn.Conv1d(4, 512, kernel_size=3, stride=1, padding=1)
            # Block 2, 4, 5, 7: ResBlock(512, 512)
            # Block 3, 6, 8: AttnBlock(512)
            # Block 9: Upsample(512) --> output shape: torch.Size([b*l, 512, 26])
            # Block 10: ResBlock(512, 256)
            # Block 11, 13: AttnBlock(256)
            # Block 12: ResBlock(256, 256)
            # Block 14: Upsample(256) --> output shape: torch.Size([b*l, 256, 52])
            # Block 15: ResBlock(256, 128)
            # Block 16: ResBlock(128, 128)
            # Block 17: GroupNorm 4, 128
            # Block 18: nn.Conv1d(128, 3, kernel_size=3, stride=1, padding=1)
            # --> output shape: torch.Size([b*l, 3, 52])
        return x



class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x



class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        # x = swish(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        # x = swish(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        # print("AttnBlock forward")
        # print("AttnBlock input shape: ", x.shape) # torch.Size([b*l, c, nj])
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_) # conv1d with num_channels c and kernel size 1
        # print("q shape: ", q.shape) # torch.Size([b*l, c, nj])
        k = self.k(h_) # conv1d with num_channels c and kernel size 1
        # print("k shape: ", k.shape) # torch.Size([b*l, c, nj])
        v = self.v(h_) # conv1d with num_channels c and kernel size 1
        # print("v shape: ", v.shape) # torch.Size([b*l, c, nj])

        # compute attention
        b, c, p = q.shape
        q = q.permute(0,2,1)   # b,p,c
        # print("permuted q shape: ", q.shape) # torch.Size([b*l, nj, c])
        w_ = torch.bmm(q,k)     # b,p,p    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_ shape: ", w_.shape) # torch.Size([b*l, nj, nj])
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # print("softmax w_ shape: ", w_.shape) # torch.Size([b*l, nj, nj])

        # attend to values
        w_ = w_.permute(0,2,1)   # b,p,p (first p of k, second of q)
        # print("permuted w_ shape: ", w_.shape) # torch.Size([b*l, nj, nj])
        h_ = torch.bmm(v,w_)     # b, c,p (p of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # print("h_ shape: ", h_.shape) # torch.Size([b*l, c, nj])

        h_ = self.proj_out(h_) # conv1d with num_channels c and kernel size 1
        # print("proj_out h_ shape: ", h_.shape) # torch.Size([b*l, c, nj])

        return x+h_
        
        # h_ = x
        # h_ = self.norm(h_)
        # q = self.q(h_)
        # k = self.k(h_)
        # v = self.v(h_)

        # # compute attention
        # b, c, h, w = q.shape
        # q = q.reshape(b, c, h*w)
        # q = q.permute(0, 2, 1)   # b,hw,c
        # k = k.reshape(b, c, h*w)  # b,c,hw
        # w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # w_ = F.softmax(w_, dim=2)

        # # attend to values
        # v = v.reshape(b, c, h*w)
        # w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b, c, h, w)

        # h_ = self.proj_out(h_)

        # return x+h_




def Normalize(in_channels):
    # print("in_channels: ", in_channels)
    return nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)