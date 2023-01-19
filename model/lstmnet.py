import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.lstm import LSTM_with_Attention, LSTM, RNN


class LSTMNet(nn.Module):
    def __init__(self, args):

        super(LSTMNet, self).__init__()
        self.args = args
        if self.args.backbone == "lstma":
            self.feature_extractor = LSTM_with_Attention(
                self.args.vocab_size,
                self.args.embedding_dim,
                self.args.hidden_dim,
                self.args.output_dim,
                n_layers=self.args.n_layers,
                use_bidirectional=self.args.use_bidirectional,
                use_dropout=self.args.use_dropout,
            )
        elif self.args.backbone == "lstm":
            self.feature_extractor = LSTM(
                self.args.vocab_size,
                self.args.embedding_dim,
                self.args.hidden_dim,
                self.args.output_dim,
                n_layers=self.args.n_layers,
                use_bidirectional=self.args.use_bidirectional,
                use_dropout=self.args.use_dropout,
            )
        else:
            print("backbone not supported")
            raise NotImplementedError
        # self.softmax = F.softmax
        self.sigmoid = F.sigmoid

    def forward(self, batch_data):
        score = self.feature_extractor(batch_data) # (batch_size, 1)
        # score = self.softmax(score, dim = 1)
        # score = self.sigmoid(score)
        return score.view(-1, 1)
