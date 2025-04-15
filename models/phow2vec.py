import os, gdown
from gensim.models import KeyedVectors
import torch
import numpy as np
from pyvi import ViTokenizer
import underthesea
from spacy.lang.vi import STOP_WORDS as VIETNAMESE_STOP_WORDS


class PhoW2VecWrapper:
    """
    Paper: PhoW2V (2020): Pre-trained Word2Vec syllable- and word-level embeddings for Vietnamese.
    Github: https://github.com/datquocnguyen/PhoW2V

    Note:
        - I (Tuan Dung) have wrapped the gensim model into a class, to make it easier to use in this PyTorch project.
        - I combine tokenization and getting word embedding into 1 class,
        please review the __call__ method.
    """

    # This is the download link given by author, not mine.
    # I use the largest model, which have the length of 300 for vector dimension.
    download_link = "https://drive.google.com/file/d/1e_zLgwkt1LmmC2rY7DOaPgjwwC-p3h0F/view?usp=drive_link"
    destination = "./downloaded_files"
    zip_file_name = "w2v.zip"
    txt_file_name = "word2vec_vi_words_300dims.txt"
    bin_file_name = "word2vec_vi_words_300dims.bin"
    npy_file_name = "word2vec_vi_words_300dims.bin.vectors.npy"

    def __init__(
        self,
        model=None,
        vector_dim=300,
        max_length=100,
        padding_value=0.0,
        padding_side="left",
    ):
        if model is None:
            if not os.path.exists(self.destination):
                os.makedirs(self.destination)
            model = self.load_model()
        self.model = model

        self.vector_dim = vector_dim
        self.max_length = max_length  # for truncation and padding
        self.padding_value = padding_value
        self.padding_side = padding_side

    def __call__(self, text):

        if type(text) == str:
            return self.get_embeddings_for_sequence(text)
        elif type(text) == list:
            list_embeddings = [self.get_embeddings_for_sequence(t) for t in text]
            # truncate long text to max_length
            list_embeddings = [
                embedding[: self.max_length] for embedding in list_embeddings
            ]
            # padding short text to max len, padding value is 0.0, on the left side
            step = 1 if self.padding_side == "left" else -1
            list_embeddings = [
                torch.cat(
                    [
                        torch.full(
                            (self.max_length - len(embedding), self.vector_dim),
                            self.padding_value,
                            dtype=torch.float,
                        ),
                        embedding,
                    ][::step]
                )
                for embedding in list_embeddings  # shape: [max_length, 300]
            ]
            tensor_embeddings = torch.stack(list_embeddings)

            # pad to min(max_length, max_len of text in batch)
            # tensor_embeddings = torch.nn.utils.rnn.pad_sequence(
            #     list_embeddings,
            #     batch_first=True,
            #     padding_side='right',
            #     padding_value=0.0,
            # )
            return tensor_embeddings

    def load_model(self):
        bin_file_path = os.path.join(self.destination, self.bin_file_name)
        if not os.path.exists(bin_file_path):
            self.download_model()
        model = KeyedVectors.load(bin_file_path)

        # test if model is loaded correctly
        assert "goá_phụ" in self.get_topk_most_similar("đàn_bà", model, topk=15)
        assert "gà" in self.get_topk_most_similar("vịt", model, topk=15)
        assert "biển" in self.get_topk_most_similar("đảo", model, topk=15)

        # load npy file and print shape vocab_size x vector_dim
        npy_file_path = os.path.join(self.destination, self.npy_file_name)
        vectors = np.load(npy_file_path)
        print("Shape of matrix word embedding:", vectors.shape)
        print("Vocab size:", vectors.shape[0])
        print("Vector dimension:", vectors.shape[1])
        return model

    # def text_preprocessing(self, text):
    #     # normalize text
    #     text = underthesea.text_normalize(text)
    #     # word segmentation
    #     text = ViTokenizer.tokenize(text.lower())
    #     # remove stop words
    #     words = text.split()
    #     words = [word for word in words if word not in VIETNAMESE_STOP_WORDS]
    #     return " ".join(words)

    def get_embeddings_for_sequence(self, text):
        # Get embeddings for a sequence of text
        # text = underthesea.text_normalize(text)
        text = ViTokenizer.tokenize(text.lower())
        words = text.split()
        embeddings = []
        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(300))
        np_array = np.array(embeddings)
        return torch.tensor(np_array).float()

    def download_model(self):
        # check if model 's zip file exists
        model_txt_path = os.path.join(self.destination, self.txt_file_name)
        if not os.path.exists(model_txt_path):
            # download zip file
            self.download_files_from_link(
                self.download_link, self.destination, self.zip_file_name
            )
            # extract zip file, output file is word2vec_vi_words_300dims.txt
            # !unzip ./downloaded_files/w2v.zip -d ./downloaded_files
            os.system(
                f"unzip {os.path.join(self.destination, self.zip_file_name)} -d {self.destination}"
            )

        # load txt file to gensim model
        model = KeyedVectors.load_word2vec_format(model_txt_path, binary=False)
        # save model to bin file
        model.save(os.path.join(self.destination, self.bin_file_name))

    @staticmethod
    def download_files_from_link(link, destination, file_name):
        file_id = link.split("/")[-2]
        url = f"https://drive.google.com/uc?id={file_id}"

        if not os.path.exists(destination):
            os.makedirs(destination)

        output = os.path.join(destination, file_name)
        print(f"Downloading {file_name} from GDrive")
        gdown.download(url, output, quiet=False)

    @staticmethod
    def get_topk_most_similar(word, model, topk=20):
        return list(dict(model.most_similar(word, topn=topk)).keys())


if __name__ == "__main__":
    w2v_model = PhoW2VecWrapper(max_length=100, padding_side="left")
    res = w2v_model(["Tôi là sinh viên trường đại học bách khoa hà nội"] * 2)
    print(res.shape)
