import torchvision
from torch import nn


class Perceptual(nn.Module):

    def __init__(self, *, network, criterion):
        super(Perceptual, self).__init__()
        self.net = network
        self.criterion = criterion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_hat, x):
        x_hat_features = self.sigmoid(self.net(x_hat))
        x_features = self.sigmoid(self.net(x))
        loss = self.criterion(x_hat_features, x_features)
        return loss


class AlexPerceptual(Perceptual):
    """
    Implements perceptual loss with a pre-trained alex net [Pihlgren et al. 2020]
    """

    def __init__(self, *, device, **kwargs):
        # Load alex net pretrained on IN1k
        alex_net = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        # Extract features after second relu activation
        # Append sigmoid layer to normalize features
        perceptual_net = alex_net.features[:5].to(device)
        # Don't record gradients for the perceptual net, the gradients will still propagate through.
        for parameter in perceptual_net.parameters():
            parameter.requires_grad = False

        super(AlexPerceptual, self).__init__(network=perceptual_net, **kwargs)


class SqueezePerceptual(Perceptual):
    """
    Implements perceptual loss with a pre-trained squeeze net
    """

    LOAD_CONFIG = {
        'device': 'cuda',
        'model_key': 'squeeze',
        'model_config': '200'
    }

    def __init__(self, **kwargs):
        from utilities.config import ConfigReader

        # Load pre-trained squeeze net
        squeeze_net = ConfigReader.load_model_from_checkpoint(**self.LOAD_CONFIG)

        # Extract features after ...
        perceptual_net = squeeze_net.features[:5]
        for parameter in perceptual_net.parameters():
            parameter.requires_grad = False
        perceptual_net.eval()

        super(SqueezePerceptual, self).__init__(network=perceptual_net, **kwargs)


if __name__ == '__main__':
    ploss_ = SqueezePerceptual(criterion=nn.MSELoss())
    print(ploss_.net)
