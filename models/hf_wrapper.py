from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from easydict import EasyDict as edict



class HuggingFaceModelWrapper(nn.Module):
    def __init__(
        self, model_name="vinai/phobert-base-v2", num_classes=3, max_length=100
    ):
        super(HuggingFaceModelWrapper, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.pooling = lambda x: x[
            :, 0, :
        ]  # just use the first token as global feature

        # my favourite MLP: linear, layer norm, gelu activation, linear
        self.classifier = self.build_classifier(
            self.backbone.config.hidden_size, num_classes
        )

        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")

        self.init_weights()
        self.print_model_info()

        # replace_bert_attention_by_flash_attention(self.backbone)
        # replace_layernorm_by_layernorm32(self)
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, texts, labels=None):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        outputs = self.backbone(**inputs)
        global_feature = self.pooling(outputs.last_hidden_state)
        logits = self.classifier(global_feature)

        return_dict = edict({"logits": logits})  # will be used in inference
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return_dict.losses = {"ce_loss": loss}  # will be used in training
        return edict(return_dict)

    def build_classifier(self, hidden_size, num_classes):
        mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )
        return mlp

    def print_model_info(self):
        print("Model name:", self.model_name)
        print("Number of classes:", self.num_classes)
        print("Max length:", self.max_length)
        print("Number of parameters:", self.count_parameters())

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.02)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

