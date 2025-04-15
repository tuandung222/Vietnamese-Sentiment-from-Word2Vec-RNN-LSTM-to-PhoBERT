import torch
from torch import nn
import torch.nn.functional as F
from easydict import EasyDict as edict


class HybridClassifer(torch.nn.Module):
    def __init__(
        self,
        word2vec_model,
        input_dim=300,
        lstm_hidden_dim=384,
        dropout=0.2,
        cnn_num_filters=300,
        cnn_filter_sizes=[3, 4, 5],
    ):
        super(HybridClassifer, self).__init__()
        self.embedding = word2vec_model

        # just one layer as in slide for speed up
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.batch_norm = nn.BatchNorm1d(lstm_hidden_dim * 2)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=lstm_hidden_dim * 2,
                    out_channels=cnn_num_filters,
                    kernel_size=fs,
                )
                for fs in cnn_filter_sizes
            ]
        )
        self.max_pools = nn.ModuleList(
            nn.AdaptiveMaxPool1d(output_size=1) for _ in cnn_filter_sizes
        )
        self.fc = nn.Linear(cnn_num_filters * len(cnn_filter_sizes), 3)
        self.dropout = nn.Dropout(dropout)

        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")
        self.init_weights()

    def forward(self, texts, labels=None):
        x = self.embedding(texts)
        x = x.to(self.device)

        # LSTM
        x, (hidden, cell) = self.lstm(x)
        # x: [batch_size, sent_len, hidden_dim * num_direct] # last layer 's output for whole sequence
        x = x.transpose(
            1, 2
        )  # Transpose to (batch_size, hidden_dim * num_direct, sent_len)
        x = self.batch_norm(x)
        x = x.transpose(
            1, 2
        )  # Transpose back to (batch_size, sent_len, hidden_dim * num_direct)

        # CNN
        x = x.transpose(-2, -1)
        conved_output = [F.relu(conv(x)) for conv in self.convs]
        pooled_output = [
            pool(conv).squeeze(-1) for conv, pool in zip(conved_output, self.max_pools)
        ]
        cat = torch.cat(pooled_output, dim=-1)
        drop_output = self.dropout(cat)
        logits = self.fc(drop_output)
        return_dict = edict({"logits": logits})
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return_dict.losses = {"ce_loss": loss}
        return edict(return_dict)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.02)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)


# hybrid_model = HybridClassifer(
#     word2vec_model=w2v_model,
#     input_dim=300,
#     lstm_hidden_dim=384,
#     dropout=0.2,
#     cnn_num_filters=300,
#     cnn_filter_sizes=[3, 4, 5, 6, 7],
# )


# hybrid_model(
#     ["Tôi là sinh viên trường đại học bách khoa hà nội"] * 3, torch.tensor([1, 0, 2])
# )
