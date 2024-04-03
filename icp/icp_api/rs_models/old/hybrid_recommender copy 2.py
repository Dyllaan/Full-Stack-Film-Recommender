import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from icp_api.models import Movie, Rating
import numpy as np

class HybridRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])
        self.buildCFModel()

    def buildCFModel(self):
        # Load the dataset from the DataFrame
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[['user_id', 'movie_id', 'rating']], reader)
        
        # Split the dataset into training and testing
        trainset, testset = train_test_split(data, test_size=0.25)  # Example split

        # Define and train the model
        self.model = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.2)
        self.model.fit(trainset)

        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f'RMSE: {rmse}')

    def get_movie_recommendations(self, user_id, n_recommendations=10):
        if not self.model:
            raise Exception("Model has not been trained. Call startup() first.")
        
        # Predict ratings for all movies the user hasn't rated yet and sort them
        predictions = []
        for movie_id in self.movies['movie_id'].unique():
            if not ((self.ratings['user_id'] == user_id) & (self.ratings['movie_id'] == movie_id)).any():
                predicted = self.model.predict(user_id, movie_id)
                predictions.append((movie_id, predicted.est))
                
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top N recommendations
        top_movie_ids = [pred[0] for pred in predictions[:n_recommendations]]
        top_movies = self.movies[self.movies['movie_id'].isin(top_movie_ids)]
        
        return top_movies[['movie_title', 'movie_genres']]