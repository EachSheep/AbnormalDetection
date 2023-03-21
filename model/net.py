import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.lstm import LSTM_with_Attention, LSTM, RNN
from model.backbone.transformer import TransformerEncoder, TransformerDecoder, Transformer, ContrastiveTransformer


class Net(nn.Module):
    def __init__(self, args):

        super(Net, self).__init__()
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
        elif self.args.backbone == "transformer":
            if self.args.criterion != "Contrastive":
                self.feature_extractor = Transformer(
                    self.args.vocab_size,
                    self.args.embedding_dim,
                    self.args.ffn_num_hiddens,
                    self.args.num_heads,
                    self.args.num_layers,
                    self.args.dropout,
                    self.args.output_dim,
                )
            else:
                self.feature_extractor = ContrastiveTransformer(
                    self.args.vocab_size,
                    self.args.embedding_dim,
                    self.args.ffn_num_hiddens,
                    self.args.num_heads,
                    self.args.num_layers,
                    self.args.dropout,
                    self.args.output_dim,
                )
        else:
            print("backbone not supported")
            raise NotImplementedError

    def forward(self, batch_data, valid_lens):

        if self.args.criterion != "Contrastive":
            score = self.feature_extractor(batch_data, valid_lens) # (batch_size, 1)
            # print("score:", score.shape)
            return score.view(-1, 1)
        
        else:
            X, score = self.feature_extractor(batch_data, valid_lens)
            return X, score.view(-1, 1)