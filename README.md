# Building an Amazon Prime content-based Movie Recommendation System
## TF-IDF, Cosine similarity, BM25, BERT

The aim of this article is to show you how to quickly create a content-based recommendation system. When you select a movie on platforms such as Amazon Prime or Netflix you may also notice that they will always show you similar movies that may be to your liking, this document shows , explains and implements three approaches to calculate those similarities using the description of each movie, the approaches are the following:

![image](https://user-images.githubusercontent.com/8701464/129777275-b09315db-ffba-4444-86cf-ec1085ac48c2.png)


## TF-IDF and Cosine Similarity

TF-IDF (term frequency-inverse document frequency) is a traditional count-based feature engineering strategy for textual data which is part of the Bag of words model, Despite is very effective for extract features from text, it is losing additional information like semantics and the context around the text. Once the raw corpus is processed by the TF-IDF, we will calculate the similarities of pairwise document using cosine similarity metric, the result of the last step is the information that the recommender needs.

## BM25

It is an improved version of TF-IDF, it will give you better relevance in the similarity than TF -IDF ->Cosine, It will not depends of the frequency of words contained in the documents and is returned more realistic results.

- TF-IDF in RED, the frequency of the words will influence the score
- BM25 in BLUE, will limit the influence of the frequency of words

![image](https://user-images.githubusercontent.com/8701464/129777298-5f219d58-6a09-420c-8cb2-502215b2fe7a.png)

![image](https://user-images.githubusercontent.com/8701464/129777306-116a0c90-21d1-4d18-ad5f-6c87a70cb797.png)


## BERT
This technique is represented by dense vectors, this means that the values of the weights matrix will have more values associated in their columns for each document, therefore, much more information in it. Internally BERT is using many encoding layers to be able to generate the dense vector, which leads to a more meaningful understanding of the text and the semantics on it. The second step in this approach is to calculate the similarities of pairwise document using cosine similarity, the result of the last step is the information that the recommender needs.


# Amazon Prime Movies Dataset
This dataset with 7261 records contains a list of all the movies streaming on the Amazon Prime platform in India.

https://www.kaggle.com/padhmam/amazon-prime-movies

# CODE
The following code represents the main class of the entire recommender, examples of how to use it will be shown further on.

```
import pandas as pd
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import BM25

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
        return self.df

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

    def get_normalized_corpus(self, tokens = False):
        n_corpus = np.vectorize(self.__normalize)        
        if tokens == True:
            norm_courpus = n_corpus(list(self.df[self.description_column]))
            return np.array([nltk.word_tokenize(d) for d in norm_corpus])            
        else:
            return n_corpus(list(self.df[self.description_column]))
            
    def get_features(self, norm_corpus):
        tf_idf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        tfidf_array = tf_idf.fit_transform(norm_corpus)
        return tfidf_array
    
    def get_vector_cosine(self, tfidf_array):
        return pd.DataFrame(cosine_similarity(tfidf_array))

    def get_bm25_weights(self, corpus):

        bm25 = BM25(corpus)
        avg_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
        weights = []
        for doc in corpus:
            scores = bm25.get_scores(doc, avg_idf)
            weights.append(scores)
            
        return pd.DataFrame(weights)
        
    def get_bert_weights(self, corpus):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        vectors = model.encode(corpus)
        weights = pd.DataFrame(cosine_similarity(vectors))
        
        return weights
    
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

```
        

## TF-IDF and Cosine Similarity
The class MovieRecommender contains all the method necessary for read datasets, clean text, and create weights based on each approach.

```
mr = MovieRecommender('archive.zip', ['Movie Name', 'Plot'], 'Movie Name', 'Plot')
df = mr.process()
mr.show_df_records(5)

```

![image](https://user-images.githubusercontent.com/8701464/129778287-fe4b22bb-d063-4047-a27b-d6cb8ba2d258.png)


The method get_normalized_corpus is cleaning the text and removing stopwords, if you pass the parameter True, it will return an array of words for each sentence.

```
norm_corpus = mr.get_normalized_corpus()
norm_corpus[:3]

```

![image](https://user-images.githubusercontent.com/8701464/129778307-93f9b161-ed7d-4054-89ba-7bcef24dd5dc.png)


The method get_features, is vectorizing the documents, converting the words to numerical values and taking into account the frequency of each word, it is applying TF-IDF

```
tfidf_array = mr.get_features(norm_corpus)
tfidf_array.shape
(7507,28013)

```

The method get_vector_cosine is returning the cosine similarity for Pairwise document similarity.

```
vector_cosine = mr.get_vector_cosine(tfidf_array)
vector_cosine.head()

```

![image](https://user-images.githubusercontent.com/8701464/129778380-0d3c895a-61d7-4be6-b2a8-ef8e0109609c.png)



This is an additional method which is useful to search options for experiments, in this case i searched “Batman” and it returns options and their ids.

```
mr.search_movies_by_term('Batman')
[(1029, 'Batman v Superman: Dawn of Justice'), (5560, 'Batman Begins')]

```

The recommendation method is used for search recommendations inside the vector of weights, notice that is receiving the vector of weights and the number of recommendations expected

```
movies_recommended  = mr.recommendation(5560, vector_cosine, 3)
print(movies_recommended)
['Pratibad' 'Bhagat Singh Ki Udeek' 'Batman v Superman: Dawn of Justice']

```
This is for search and check the description of movies:

```
df[df['Movie Name'] == 'Pratibad' ].values

```
![image](https://user-images.githubusercontent.com/8701464/129778447-e949aba8-d73c-4951-a5ce-ff57a20dd29a.png)


```
df[df['Movie Name'] == 'Bhagat Singh Ki Udeek' ].values

```

![image](https://user-images.githubusercontent.com/8701464/129778467-e1940af3-f17c-4796-864a-a7f44624c958.png)


```
df[df['Movie Name'] == 'Batman Begins' ].values

```

![image](https://user-images.githubusercontent.com/8701464/129778476-595fe022-c850-40e1-81ed-212170421d38.png)


# BM25
BM25 es expecting receive the documents as a tokens

```
norm_corpus_tokens = mr.get_normalized_corpus(True)
norm_corpus_tokens[:3]

```

![image](https://user-images.githubusercontent.com/8701464/129778015-a40cc35b-7b83-4067-b46a-53a51d09be02.png)


```
wts = mr.get_bm25_weights(norm_corpus_tokens)
bm25_wts_df = pd.DataFrame(wts)
bm25_wts_df.head()

```

![image](https://user-images.githubusercontent.com/8701464/129778042-3123763f-d2d4-45aa-9232-13875386fe03.png)


```
movies_recommended  = mr.recommendation(5560, bm25_wts_df, 3)
print(movies_recommended)
['The Dark Knight' 'Pratibad' 'Akrandhana']

```

## BERT


```
wts_df = mr.get_bert_weights(norm_corpus)
wts_df.head()

```

```
movies_recommended  = mr.recommendation(5560, wts_df, 3)
print(movies_recommended)
[‘The Dark Knight’ ‘Dune’ ‘Wake Of Death’ ]

```

```
df[df['Movie Name'] == 'Dune' ].values

```

![image](https://user-images.githubusercontent.com/8701464/129777897-5ade1d86-44ee-464d-80c2-8093e4967d5c.png)


# Summary
## Batman Begins
In the wake of his parents murder, disillusioned industrial heir Bruce Wayne travels the world seeking the means to fight injustice.

- So dramatic! We already know that Batman is too weird, but checking the words involved in the description “fight injustice” is totally sufficient for techniques based on semantics.
- All the movies recommended by the three approaches seem correct to me, there is someone involved in the fight for injustice.
- The third recommendation given by the BM25 approach does not seem correct to me.
- The recommendations given by BERT seem very natural and logical to me, you can notice the absence of the words mentioned in point 1, here you can see that the recommendations are based on the semantics of the text and not only on the frequency of the words.
- To conclude, vectorization techniques that generate dense vectors are more robust and more sensitive to detecting natural language.

![image](https://user-images.githubusercontent.com/8701464/129777833-5f98c15f-a699-484c-92e3-242a9ca27ec8.png)




