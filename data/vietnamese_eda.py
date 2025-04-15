"""
    Copied and modifed from <https://github.com/dsfsi/textaugment/>
    This origin module is an implementation of the original EDA algorithm:
        - EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
        - https://arxiv.org/abs/1901.11196
    I modify this module has been modified to support Vietnamese language.
"""

from spacy.lang.vi import STOP_WORDS as VIETNAMESE_STOP_WORDS
import random


import json
import random

from random import shuffle
import os
import wget
import underthesea

from spacy.lang.vi import STOP_WORDS as VIETNAMESE_STOP_WORDS

UNDERTHESEA_NORMALIZER = underthesea.text_normalize
UNDERTHESEA_NORMALIZER.normalize = lambda x: underthesea.text_normalize(x)
from pyvi import ViTokenizer as PYVI_SEGMENTER


class VietnameseWordnet:

    download_folder = "downloaded_files"
    vi_wordnet_path = f"{download_folder}/word_net_vi.json"
    link = "https://raw.githubusercontent.com/sonlam1102/text_augmentation_vietnamese/main/word_net_vi.json"

    def __init__(self) -> None:
        os.makedirs(self.download_folder, exist_ok=True)
        wget.download(self.link, self.vi_wordnet_path)
        with open(self.vi_wordnet_path) as f:
            vi_wordnet_dictionary = json.load(f)
        self.vi_wordnet_dictionary = self.segment_dictionary(vi_wordnet_dictionary)

    def segment_dictionary(self, dictionary):
        return_dict = {
            "_".join(word.split()): [
                "_".join(synset.split()) for synset in list_synsets
            ]
            for word, list_synsets in dictionary.items()
        }
        return return_dict

    def synsets(self, word):
        return self.vi_wordnet_dictionary.get(word, [])


class VietnameseEDATransform:
    def __init__(
        self,
        corpus=VietnameseWordnet(),
        stop_words=VIETNAMESE_STOP_WORDS,
        word_segmenter=PYVI_SEGMENTER,
        text_normalizer=UNDERTHESEA_NORMALIZER,
        p_to_remove_stopwords=0.1,
        alpha_sr=0.1,
        alpha_ri=0.1,
        alpha_rs=0.1,
        p_rd=0.1,
        num_aug=1,
        do_unsegment_output=True,
    ) -> None:
        self.corpus = corpus
        self.stop_words = stop_words
        self.word_segmenter = word_segmenter
        self.text_normalizer = text_normalizer
        self.p_to_remove_stopwords = p_to_remove_stopwords
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug
        # de-transform the output 'Ha_Noi' to 'Ha Noi'
        self.unsegment_output = do_unsegment_output

        assert "tokenize" in dir(
            self.word_segmenter
        ), "word_segmenter must have tokenize method"
        assert "normalize" in dir(
            self.text_normalizer
        ), "text_normalizer must have normalize method"

    def __call__(self, sentence):
        # this fuction return list of original sentence and augmented sentences
        try:
            return self.eda(
                sentence,
                self.p_to_remove_stopwords,
                self.alpha_sr,
                self.alpha_ri,
                self.alpha_rs,
                self.p_rd,
                self.num_aug,
                self.word_segmenter,
                self.text_normalizer,
                self.unsegment_output,
            )
        except Exception as e:
            return [sentence]

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(
            set([word for word in words if word not in self.stop_words])
        )
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word, self.corpus)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [
                    synonym if word == random_word else word for word in new_words
                ]
                # print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n:  # only replace up to n words
                break

        # this is stupid but we need it, trust me
        sentence = " ".join(new_words)
        new_words = sentence.split(" ")

        return new_words

    def get_synonyms(self, word, corpus):
        return corpus.synsets(word)

    def random_deletion(self, words, p):

        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        # randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        # if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = (
            new_words[random_idx_2],
            new_words[random_idx_1],
        )
        return new_words

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word, self.corpus)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    def eda(
        self,
        sentence,
        p_to_remove_stopwords=0.5,
        alpha_sr=0.1,
        alpha_ri=0.1,
        alpha_rs=0.1,
        p_rd=0.1,
        num_aug=9,
        word_segmenter=PYVI_SEGMENTER,
        text_normalizer=UNDERTHESEA_NORMALIZER,
        un_segment_output=True,
    ):

        # NOTE: add these line for Vietnamese
        sentence = text_normalizer.normalize(sentence)
        sentence = word_segmenter.tokenize(sentence)

        words = sentence.split(" ")
        words = [word for word in words if word != ""]

        # NOTE: add these line for remove Vietnamese stop words
        p_gen_stopwords = random.uniform(0, 1)
        if p_gen_stopwords < p_to_remove_stopwords:
            words = [word for word in words if word not in VIETNAMESE_STOP_WORDS]

        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1

        # sr
        if alpha_sr > 0:
            n_sr = max(1, int(alpha_sr * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sentences.append(" ".join(a_words))

        # ri
        if alpha_ri > 0:
            n_ri = max(1, int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sentences.append(" ".join(a_words))

        # rs
        if alpha_rs > 0:
            n_rs = max(1, int(alpha_rs * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words, n_rs)
                augmented_sentences.append(" ".join(a_words))

        # rd
        if p_rd > 0:
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, p_rd)
                augmented_sentences.append(" ".join(a_words))

        # NOTE: modify this line for Vietnamese
        # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
        augmented_sentences = [
            text_normalizer.normalize(sentence) for sentence in augmented_sentences
        ]

        shuffle(augmented_sentences)

        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [
                s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
            ]

        # append the original sentence
        augmented_sentences.append(sentence)

        # NOTE: add this line for Vietnamese
        if un_segment_output:
            augmented_sentences = [
                augmented_sentence.replace("_", " ")
                .replace(" ,", ",")
                .replace(" .", ".")
                for augmented_sentence in augmented_sentences
            ]
        return augmented_sentences
