# coding: utf-8
import MeCab
import numpy as np
import glob
from gensim.models.word2vec import Word2Vec
from utils.cos_sim import cos_sim

# load model
model_path = 'model/wiki.model'
model = Word2Vec.load(model_path)

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

test_folder = glob.glob('test/*')

sentence_vectors = []

for test_file in test_folder:
    word_number = 0
    sentence_vector = np.zeros(300)
    f = open(test_file, 'r')
    for line in f:
        node = mecab.parseToNode(line)
        while node:
            # get word
            word = node.surface
            # get part of speech(POS)
            pos = node.feature.split(',')[1]
            if pos == '固有名詞' or pos == '一般':
                try:
                    vector_value = model.wv[word]
                    word_number += 1
                    sentence_vector += vector_value
                except KeyError:
                    pass

            # go to next word
            node = node.next
    sentence_vectors.append(sentence_vector/word_number)

print(cos_sim(sentence_vectors[0], sentence_vectors[1]))
