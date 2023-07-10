import torch.nn as nn


class EmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(20, 32, 3, 1),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.AvgPool1d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(),
                                    nn.AvgPool1d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, 1),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(),
                                    nn.AvgPool1d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 64, 3, 1),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(),
                                    nn.AvgPool1d(2, 2))
        self.conv5 = nn.Sequential(nn.Conv1d(64, 32, 3, 1),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.AvgPool1d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(32, 16),
                                  nn.Dropout(0.5),
                                  nn.LeakyReLU())
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc1(x)
        x = self.fc2(x)

        return x