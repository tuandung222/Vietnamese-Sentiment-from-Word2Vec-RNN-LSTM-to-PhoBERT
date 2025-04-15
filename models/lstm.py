from torch.nn.modules import dropout
import torch
from torch import nn
from easydict import EasyDict as edict

class LSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        word2vec_model,
        input_dim=300,
        hidden_dims=[384, 384, 384],
        output_dim=3,
        n_layers=3,
        bidirectional=True,
        dropout=0.2,
    ):
        super(LSTMClassifier, self).__init__()
        self.embedding = word2vec_model

        num_direct = 2 if bidirectional else 1
        # hidden_dims is vector dim of single direction, output_dim of lstm is hidden_dim * num_direct

        list_in_lstm_dims = [input_dim] + [
            hidden_dims[i] * num_direct for i in range(len(hidden_dims) - 1)
        ]
        list_out_lstm_dims = hidden_dims

        self.lstm_chain = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=list_in_lstm_dims[i],
                    hidden_size=list_out_lstm_dims[i],
                    num_layers=1,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=True,
                )
                for i in range(n_layers)
            ]
        )
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(list_out_lstm_dims[-1] * num_direct, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.02)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def forward(self, texts, labels=None):
        x = self.embedding(texts)
        x = x.to(self.device)  # [batch_size, sent_len, input_dim]

        for lstm in self.lstm_chain:
            x, (hidden, cell) = lstm(x)

        # num_direct = 2 if bidirectional else 1
        # x: [batch_size, sent_len, hidden_dim * num_direct] # last layer 's output for whole sequence
        # hidden: [num_layer * num_direct, batch_size, hidden_dim] # last hidden state of all layers
        # cell: [num_layer * num_direct, batch_size, hidden_dim] # last cell state of all layers

        # take the last hidden state of the last layer as global feature
        # the input have been padded to the right, so we can take the last hidden state as global feature
        global_feature = x[:, -1, :]  # [batch_size, hidden_dim * num_direct]

        dropout_output = self.dropout(
            global_feature
        )  # [batch_size, hidden_dim * num_direct]
        logits = self.fc(dropout_output)
        return_dict = edict({"logits": logits})
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return_dict.losses = {"ce_loss": loss}
        return edict(return_dict)
