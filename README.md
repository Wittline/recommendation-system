# Building an Amazon Prime content-based Movie Recommendation System
## TF-IDF, Cosine similarity, BM25, BERT

The aim of this article is to show you how to quickly create a content-based recommendation system. When you select a movie on platforms such as Amazon Prime or Netflix you may also notice that they will always show you similar movies that may be to your liking, this document shows , explains and implements three approaches to calculate those similarities using the description of each movie, the approaches are the following:

## TF-IDF and Cosine Similarity

TF-IDF (term frequency-inverse document frequency) is a traditional count-based feature engineering strategy for textual data which is part of the Bag of words model, Despite is very effective for extract features from text, it is losing additional information like semantics and the context around the text. Once the raw corpus is processed by the TF-IDF, we will calculate the similarities of pairwise document using cosine similarity metric, the result of the last step is the information that the recommender needs.

## BM25

It is an improved version of TF-IDF, it will give you better relevance in the similarity than TF -IDF ->Cosine, It will not depends of the frequency of words contained in the documents and is returned more realistic results.

- TF-IDF in RED, the frequency of the words will influence the score
- BM25 in BLUE, will limit the influence of the frequency of words

## BERT
This technique is represented by dense vectors, this means that the values of the weights matrix will have more values associated in their columns for each document, therefore, much more information in it. Internally BERT is using many encoding layers to be able to generate the dense vector, which leads to a more meaningful understanding of the text and the semantics on it. The second step in this approach is to calculate the similarities of pairwise document using cosine similarity, the result of the last step is the information that the recommender needs.


# Amazon Prime Movies Dataset
This dataset with 7261 records contains a list of all the movies streaming on the Amazon Prime platform in India.

# CODE
The following code represents the main class of the entire recommender, examples of how to use it will be shown further on.

TF-IDF and Cosine Similarity
The class MovieRecommender contains all the method necessary for read datasets, clean text, and create weights based on each approach.
mr = MovieRecommender('archive.zip', ['Movie Name', 'Plot'], 'Movie Name', 'Plot')
df = mr.process()
mr.show_df_records(5)

The method get_normalized_corpus is cleaning the text and removing stopwords, if you pass the parameter True, it will return an array of words for each sentence.
norm_corpus = mr.get_normalized_corpus()
norm_corpus[:3]

The method get_features, is vectorizing the documents, converting the words to numerical values and taking into account the frequency of each word, it is applying TF-IDF
tfidf_array = mr.get_features(norm_corpus)
tfidf_array.shape
(7507,28013)
The method get_vector_cosine is returning the cosine similarity for Pairwise document similarity.
vector_cosine = mr.get_vector_cosine(tfidf_array)
vector_cosine.head()

This is an additional method which is useful to search options for experiments, in this case i searched “Batman” and it returns options and their ids.
mr.search_movies_by_term('Batman')
[(1029, 'Batman v Superman: Dawn of Justice'), (5560, 'Batman Begins')]
The recommendation method is used for search recommendations inside the vector of weights, notice that is receiving the vector of weights and the number of recommendations expected
movies_recommended  = mr.recommendation(5560, vector_cosine, 3)
print(movies_recommended)
['Pratibad' 'Bhagat Singh Ki Udeek' 'Batman v Superman: Dawn of Justice']
This is for search and check the description of movies:
df[df['Movie Name'] == 'Pratibad' ].values

df[df['Movie Name'] == 'Bhagat Singh Ki Udeek' ].values

df[df['Movie Name'] == 'Batman Begins' ].values

BM25
BM25 es expecting receive the documents as a tokens
norm_corpus_tokens = mr.get_normalized_corpus(True)
norm_corpus_tokens[:3]

wts = mr.get_bm25_weights(norm_corpus_tokens)
bm25_wts_df = pd.DataFrame(wts)
bm25_wts_df.head()

movies_recommended  = mr.recommendation(5560, bm25_wts_df, 3)
print(movies_recommended)
['The Dark Knight' 'Pratibad' 'Akrandhana']
BERT
wts_df = mr.get_bert_weights(norm_corpus)
wts_df.head()
movies_recommended  = mr.recommendation(5560, wts_df, 3)
print(movies_recommended)
[‘The Dark Knight’ ‘Dune’ ‘Wake Of Death’ ]
df[df['Movie Name'] == 'Dune' ].values

Check out my GitHub repository for more information about the code.
Summary
Batman Begins
In the wake of his parents murder, disillusioned industrial heir Bruce Wayne travels the world seeking the means to fight injustice.
So dramatic! We already know that Batman is too weird, but checking the words involved in the description “fight injustice” is totally sufficient for techniques based on semantics.
All the movies recommended by the three approaches seem correct to me, there is someone involved in the fight for injustice.
The third recommendation given by the BM25 approach does not seem correct to me.
The recommendations given by BERT seem very natural and logical to me, you can notice the absence of the words mentioned in point 1, here you can see that the recommendations are based on the semantics of the text and not only on the frequency of the words.
To conclude, vectorization techniques that generate dense vectors are more robust and more sensitive to detecting natural language.

Results: TF-IDF, BM25, BERT







