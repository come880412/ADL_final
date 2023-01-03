import torch.nn as nn
import torch

from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast

class RS_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weighted_sum = nn.Linear(12, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )

    def forward(self, embedding):
        embedding = self.weighted_sum(embedding[:, :, 1:]).squeeze(-1)
        # embedding = embedding[:, :, -1]

        output = self.classifier(embedding)
        return output