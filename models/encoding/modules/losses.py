import torch.nn as nn
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, reconstruction, split='train'):
        mse_loss = self.mse(input, reconstruction)
        mse_loss = torch.mean(mse_loss)

        # log = {
        #     "{}/mse_loss".format(split): mse_loss.clone().detach().cpu().numpy(),
        # }

        return mse_loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input, reconstruction, split='train'):
        l1_loss = self.l1(input, reconstruction)
        l1_loss = torch.mean(l1_loss)

        # log = {
        #     "{}/l1_loss".format(split): l1_loss.clone().detach().cpu().numpy(),
        # }

        return l1_loss