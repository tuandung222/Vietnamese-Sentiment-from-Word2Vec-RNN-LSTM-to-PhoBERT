import pandas as pd
import datasets
import underthesea
from pyvi import ViTokenizer
from spacy.lang.vi import STOP_WORDS as VIETNAMESE_STOP_WORDS
import random
import torch
# from .vietnamese_eda import VietnameseEDATransform


class DatasetBuilder:

    train_file = "annotations/vlsp_sentiment_train.csv"
    test_file = "annotations/vlsp_sentiment_test.csv"

    def __init__(self, train_augment_transform=lambda x: x):
        self.train_augment_transform = train_augment_transform

    def read_data(self, file_path, is_train=True):
        # remove header, two columns with name 'Class' and 'Data'
        data = pd.read_csv(
            file_path, delimiter="\t", header=None, names=["label", "text"]
        )
        data = data[1:]
        data = datasets.Dataset.from_pandas(data)
        if is_train:
            # transform: just get a random EDA transform if exists, else original text
            transform = lambda list_text: [
                random.choice(self.train_augment_transform(x) or [x]) for x in list_text
            ]
        else:
            transform = lambda x: x
        data = data.map(lambda x: {"label": int(x["label"]) + 1, "text": (x["text"])})
        # data.set_transform(lambda x: {"label": x["label"], "text": transform(x["text"])})
        data = data.cast_column(
            "label",
            datasets.ClassLabel(
                num_classes=3, names=["negative", "neutral", "positive"]
            ),
        )
        return data

    def get_transform(self, is_train):
        if is_train:
            aug_transform = lambda list_text: [
                random.choice(self.train_augment_transform(x)) for x in list_text
            ]
        else:
            aug_transform = lambda x: x
        return lambda x: {"label": (x["label"]), "text": aug_transform(x["text"])}

    def build(self):
        train_table = self.read_data(self.train_file, is_train=True)
        # split train_table to train and validation
        train_val_dict = train_table.train_test_split(
            test_size=0.1, stratify_by_column="label", seed=44
        )
        train_table = train_val_dict["train"]
        val_table = train_val_dict["test"]

        test_table = self.read_data(self.test_file, is_train=False)

        train_table.set_transform(self.get_transform(is_train=True))
        val_table.set_transform(self.get_transform(is_train=False))
        test_table.set_transform(self.get_transform(is_train=False))

        return train_table, val_table, test_table  # can be use as torch.Dataset
    
    @staticmethod
    def collate_fn(list_examples):
        # list_examples: list of examples, each example is a dict
        # list_examples[0] = {"label": tensor, "text": text}
        # ...
        # list_examples[n] = {"label": tensor, "text": text}
        # return: dict of batched examples
        labels = torch.Tensor([example["label"] for example in list_examples]).long()

        # weird implementation that combine model and tokenizer into 1
        texts = [example["text"] for example in list_examples]
        return {"labels": labels, "texts": texts}

if __name__ == "__main__":
    builder = DatasetBuilder()
    train_table, val_table, test_table = builder.build()
