import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """docstring for ParserModel"""

    def __init__(self, embeddings, n_features = 36, hidden_size = 200, n_classes = 3, dropout_prob = 0.3):
        super(ParserModel, self).__init__()

        self.embed_size = embeddings.shape[0]
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0],embedding.shape[1])
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob

        self.embed_to_hidden = nn.Linear(self.embed_size*self.n_features, self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)

    def embedding_lookup(self, t):
        """t is the input tokens (integer)"""
        embedding = self.pretrained_embeddings(t)
        print("shape of input tokens in 'embedding_lookup':",t.shape)
        x = embedding.view(t.shape[0], self.n_features*self.embed_size)

        return x

    def forward(self, x):
        x = embedding_lookup(x)
        x = F.relu(self.embed_to_hidden(x))
        x = self.dropout(x)
        x = self.hidden_to_logits(x)

        return x
