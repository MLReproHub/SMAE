from torch import nn


class Probe(nn.Module):
    """
    Probe for classification.
    """

    def __init__(self, *, d_model, normalize, num_classes, device="cuda"):
        super(Probe, self).__init__()
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(d_model, affine=False)
        self.head = nn.Linear(d_model, num_classes)
        self.to(device)

    def forward(self, x):
        if self.normalize:
            x = self.bn(x)
        y = self.head(x)
        return y
