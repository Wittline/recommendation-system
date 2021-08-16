import math
from six import iteritems
from six.moves import xrange

class BM25(object):

    def __init__(self, corpus):

        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.initialize()

    def initialize(self):

        for document in self.corpus:
            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf, k1 = 2.5, b = 0.85, e = 0.2):

        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else e * average_idf
            score += (idf * self.f[index][word] * (k1 + 1)
                      / (self.f[index][word] + k1 * (1 - b + b * self.doc_len[index] / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores