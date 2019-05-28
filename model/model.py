# coding: utf-8
from gensim.models import word2vec

corpus = word2vec.Text8Corpus('./model/wiki_wakati.txt')

model = word2vec.Word2Vec(corpus, size=300, min_count=5, window=15)
model.save("./wiki.model")
