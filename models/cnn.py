from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(
        self, word2vec_model, input_dim, num_filters, filter_sizes, output_dim, dropout
    ):
        super(CNNClassifier, self).__init__()
        self.embedding = word2vec_model

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim, out_channels=num_filters, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )
        self.max_pools = nn.ModuleList(
            nn.AdaptiveMaxPool1d(output_size=1) for _ in filter_sizes
        )  # make (B, C, L) to (B, C, 1), use adaptive for not caring about length of input
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
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
        # x is text or len of text
        # this is weird because I combined model and tokenizer into 1
        x = self.embedding(texts)  # [batch_size, sent_len, emb_dim]
        x = x.transpose(-2, -1)  # [batch_size, emb_dim, sent_len]
        x = x.to(self.device)
        # consider emb_dim as input channel
        conved_output = [
            F.relu(conv(x)) for conv in self.convs
        ]  # list of tensor shaped [batch_size, num_filter, sent_len - filter_sizes[n] + 1]
        pooled_output = [
            pool(conv).squeeze(-1) for conv, pool in zip(conved_output, self.max_pools)
        ]  # list of tensor shaped [batch_size, num_filter]
        cat = torch.cat(
            pooled_output, dim=-1
        )  # [batch_size, num_filter * len(filter_sizes)]
        drop_output = self.dropout(cat)  # [batch_size, num_filter * len(filter_sizes)]
        logits = self.fc(drop_output)  # [batch_size, output_dim]

        # output should be wrapped in edict, for multi-way attribute-accessing
        return_dict = edict({"logits": logits})  # logits will use in inference
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return_dict.losses = {"ce_loss": loss}  # losses will use in training

        return edict(return_dict)

if __name__ == "__main__":
    # test in root folder
    from phow2vec import PhoW2VecWrapper
    w2v_model = PhoW2VecWrapper(max_length=100, padding_side="left")
    cnn_model = CNNClassifier(
            word2vec_model=w2v_model,
            input_dim=300,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            output_dim=3,
            dropout=0.1,
    )

    cnn_model(
        ["Tôi là sinh viên trường đại học bách khoa hà nội"] * 3, torch.tensor([1, 0, 2])
    )