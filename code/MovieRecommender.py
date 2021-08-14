import pandas as pd
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:

    def __init__(self, filename, columns, t_column, d_column):
        self.filename = filename
        self.columns = columns
        self.title_column = t_column
        self.description_column = d_column
        self.df = None

    def process(self, show=True):
        self.df = pd.read_csv(self.filename)
        self.df = self.df[self.columns]
        self.df[self.description_column].fillna('', inplace=True)
        self.df[self.description_column] = self.df[self.title_column] + '. ' +  self.df[self.description_column].map(str)
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

    def show_df_records(self, n = 5):
        return self.df.head(n)

    def show_info_details(self):
        return self.df.info()

    def __normalize(self, d):
        stopwords = nltk.corpus.stopwords.words('english')
        d = re.sub(r'[^a-zA-Z0-9\s]', '', d, re.I|re.A)
        d = d.lower().strip()
        tks = nltk.word_tokenize(d)
        f_tks = [t for t in tks if t not in stopwords]
        return ' '.join(f_tks)

    def get_normalized_corpus(self):
        n_corpus = np.vectorize(self.__normalize)
        return n_corpus(list(self.df[self.description_column]))

    def get_features(self, norm_corpus):
        tf_idf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        tfidf_array = tf_idf.fit_transform(norm_corpus)
        return tfidf_array
    
    def get_vector_cosine(self, tfidf_array):
        return pd.DataFrame(cosine_similarity(tfidf_array))
    
    def search_movies_by_term(self, term='movie'):
        movies = self.df[self.title_column].values
        possible_options = [(i, movie) for i, movie in enumerate(movies) for word in movie.split(' ') if word == term]
        return possible_options
    
    def recommendation(self, index, vector, n):
        similarities = vector.iloc[index].values
        similar_indices = np.argsort(-similarities)[1:n + 1]
        movies = self.df[self.title_column].values
        similar_movies =  movies[similar_indices]
        return similar_movies