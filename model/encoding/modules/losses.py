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