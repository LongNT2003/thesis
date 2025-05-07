from torch import nn


class EmbeddingHead(nn.Module):
    def __init__(self, embedding_size, in_features, dropout=0.2):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(in_features)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, embedding_size, bias=True)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)  # Output: (N, embedding_size)
        return nn.functional.normalize(x, dim=-1)  # Chuẩn hóa theo chiều feature
