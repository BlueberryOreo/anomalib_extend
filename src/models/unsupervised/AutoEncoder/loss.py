from torch import nn

class AutoEncoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l2_loss = nn.modules.loss.MSELoss()

    def forward(self, input_image, reconstruction):
        """Compute the loss over a batch for the DRAEM model."""
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        return l2_loss_val