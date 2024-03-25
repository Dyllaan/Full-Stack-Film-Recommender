import pandas as pd
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')

# Preprocess and combine movie tags and genres
movies['genres'] = movies['genres'].str.replace('|', ' ')
tags['tag'] = tags.groupby('movieId')['tag'].transform(lambda x: ' '.join(x))
tags = tags[['movieId', 'tag']].drop_duplicates()
movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')
movies_with_tags['combined_features'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag'].fillna('')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# For SVD++
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Set up cross-validation
kf = KFold(n_splits=5)

# Build and evaluate the SVD++ model using cross-validation
svdpp = SVDpp()
cross_val_results = cross_validate(svdpp, data, measures=['RMSE'], cv=kf, verbose=True)

# Create a mapping between movie IDs and indices
movie_indices = pd.Series(movies_with_tags.index, index=movies_with_tags['movieId'])
# Modify the get_recommendations function to use this mapping
def get_recommendations(movie_id, movie_indices, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the movie_id
    idx = movie_indices[movie_id]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_with_tags['movieId'].iloc[movie_indices].tolist()

# Modify the hybrid_recommendations function accordingly
def hybrid_recommendations(user_id, movie_id, movie_indices, svdpp, ratings, svdpp_weight=0.5, cbf_weight=0.5):
    # Calculate user bias
    user_avg = ratings[ratings['userId'] == user_id]['rating'].mean()
    overall_avg = ratings['rating'].mean()
    user_bias = user_avg - overall_avg

    # Get CBF and SVD++ recommendations
    cbf_recommendations = get_recommendations(movie_id, movie_indices)
    svdpp_predictions = [svdpp.predict(user_id, cbf_movie_id).est for cbf_movie_id in cbf_recommendations]

    combined_scores = []
    for i, cbf_movie_id in enumerate(cbf_recommendations):
        idx = movie_indices[cbf_movie_id]
        # Adjust SVD++ predictions by subtracting user bias
        adjusted_svdpp_score = svdpp_predictions[i] - user_bias
        combined_score = svdpp_weight * adjusted_svdpp_score + cbf_weight * (1 - cosine_sim[movie_indices[movie_id]][idx])
        combined_scores.append((cbf_movie_id, combined_score))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = [rec[0] for rec in combined_scores[:10]]
    return top_recommendations

# Example usage - Note that this will not work as expected because 'svdpp' is not a trained model here. You need to train the model on the full dataset for actual predictions.
user_id_example = 1
movie_id_example = 1  # Toy Story
hybrid_recommendations_example = hybrid_recommendations(user_id_example, movie_id_example, movie_indices, svdpp, ratings)
print("Hybrid Recommendations:", hybrid_recommendations_example)