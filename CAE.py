from torch import nn



class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 256, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv1d(256, 128, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv1d(128, 128, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv1d(128, 64, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv1d(64, 64, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.Conv1d(64, 32, 9, stride=2))

        self.decoder = nn.Sequential(nn.ConvTranspose1d(32, 64, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(64, 64, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(64, 128, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(128, 128, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(128, 256, 9, stride=2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(256, 1, 9, stride=2),
                                     nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x