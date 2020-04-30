import torch
from torch import nn


class Linearclassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.classifier = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    @classmethod
    def from_autoencoder(cls, path):
        model = cls()
        decode_layers = ["decoder.0.weight", "decoder.0.bias", "decoder.2.weight", "decoder.2.bias"]
        m_state = torch.load(path)
        for layer in decode_layers:
            m_state.pop(layer)
        for name, layer in model.state_dict().items():
            if name not in m_state:
                m_state[name] = layer
        model.load_state_dict(m_state)
        return model


class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
