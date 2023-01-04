import torch.nn as nn
import torch

from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast

class RS_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weighted_sum = nn.Linear(12, 1)

        self.classifier = nn.Sequential(
            # nn.Linear(95, 128),
            nn.Linear(95, num_classes),
            # nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # embedding = self.weighted_sum(embedding[:, :, 1:]).squeeze(-1)
        # embedding = embedding[:, :, -1]
        # print("x",x)
        output = self.classifier(x)
        return output