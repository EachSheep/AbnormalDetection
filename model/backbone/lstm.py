"""https://github.com/slaysd/pytorch-sentiment-analysis-classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, use_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class LSTM(nn.Module):
    """
    Args:
        vocab_size (int): size of the vocabulary，输入的词的大小，有多少种不同的页面就有多少种不同的词
        embedding_dim (int): size of the word embeddings，词向量的维度
        hidden_dim (int): size of the hidden state of the LSTM，隐藏层的维度，也是输出向量的维度
        output_dim (int): size of the output of the LSTM
        n_layers (int): number of layers in the LSTM，recurrent layer的数量，默认1
        use_bidirectional (bool): whether to use bidirectional LSTM
        use_dropout (bool): whether to use dropout
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, use_dropout=False):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x, valid_lens):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                        dim=1))
        return self.fc(hidden.squeeze(0))


class LSTM_with_Attention(nn.Module):
    """
    Args:
        vocab_size (int): size of the vocabulary，输入的词的大小，有多少种不同的页面就有多少种不同的词
        embedding_dim (int): size of the word embeddings，词向量的维度
        hidden_dim (int): size of the hidden state of the LSTM，隐藏层的维度，也是输出向量的维度
        output_dim (int): size of the output of the LSTM
        n_layers (int): number of layers in the LSTM，recurrent layer的数量，默认1
        use_bidirectional (bool): whether to use bidirectional LSTM
        use_dropout (bool): whether to use dropout
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=use_bidirectional,
            dropout=0.5 if use_dropout else 0.
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x, valid_lens):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)

        return self.fc(attn_output.squeeze(0))


class CNN_LSTM(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
