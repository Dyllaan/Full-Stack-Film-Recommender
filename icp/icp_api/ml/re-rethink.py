import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import cross_validate
import numpy as np

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.2)
# Run 5-fold cross-validation and print results.
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
model.fit(trainset)

test_predictions = model.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(test_predictions)
print(f'RMSE: {rmse}')


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Function to build a user profile
def build_user_profile(user_id, ratings, tfidf_matrix):
    # Filter movies rated by the user
    rated_movies = ratings[ratings['userId'] == user_id]
    # Filter movies with a rating above a certain threshold, e.g., 4.0
    high_rated_movies = rated_movies[rated_movies['rating'] >= 4.0]
    # Get the indices of these movies in the TF-IDF matrix
    indices = [movies.index[movies['movieId'] == movie_id].tolist()[0] for movie_id in high_rated_movies['movieId']]
    # Aggregate the TF-IDF vectors of these movies
    user_profile = np.mean(tfidf_matrix[indices], axis=0)
    # Convert to numpy array if it's not already
    if isinstance(user_profile, np.matrix):
        user_profile = np.asarray(user_profile)
    return user_profile

# Function to recommend movies based on user profile
def cbf_recommendations():
    user_profile = build_user_profile(user_id, ratings, tfidf_matrix)
    # Ensure user_profile is in the correct format
    user_profile = user_profile.reshape(1, -1)  # Reshape for compatibility with cosine_similarity
    # Calculate cosine similarity between user profile and all item profiles
    cos_similarity = cosine_similarity(user_profile, tfidf_matrix)
    # Get the top N recommendations
    top_indices = np.argsort(cos_similarity[0])[-top_n:]
    recommended_movies = movies.iloc[top_indices]
    return recommended_movies
    
# Function to predict ratings for all movies a user hasn't rated
def cf_recommendations():
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].unique()
    unrated_movies = movies[~movies['movieId'].isin(rated_movie_ids)]
    predictions = []
    for _, row in unrated_movies.iterrows():
        movie_id = row['movieId']
        predicted_rating = model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    return pd.DataFrame(predictions, columns=['movieId', 'predictedRating']).sort_values('predictedRating', ascending=False)

df_movies=movies
def hybrid_content_svd_model():
    """
    hydrid the functionality of content based and svd based model to recommend user top 10 movies. 
    :param userId: userId of user
    :return: list of movies recommended with rating given by svd model
    """
    # Get the top N recommendations from content-based filtering
    content_based_recommendations = cbf_recommendations()
    # Get the top N recommendations from collaborative filtering
    collaborative_filtering_recommendations = cf_recommendations()
    # Merge the two recommendation lists
    recommendations = pd.merge(content_based_recommendations, collaborative_filtering_recommendations, on='movieId')
    # Calculate a hybrid score based on the average of the two recommendation scores
    recommendations['hybrid_score'] = (recommendations['predictedRating'] + recommendations['rating']) / 2
    # Sort the recommendations by the hybrid score
    recommendations = recommendations.sort_values('hybrid_score', ascending=False)
    return recommendations
        

# Example usage
user_id = 1
top_n = 10
predicted = hybrid_content_svd_model()
top_n_recommendations = predicted.head(top_n)
print(top_n_recommendations)