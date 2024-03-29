import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from icp_api.models import Movie, Rating

class HybridRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])
        self.best_model_params = None
        self.best_rmse = float('inf')
        self.best_model = None
        self.build_best_CF_model()

    def build_CF_model(self, n_factors, n_epochs, lr_all, reg_all):
        # Load the dataset from the DataFrame
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[['user_id', 'movie_id', 'rating']], reader)
        
        # Split the dataset into training and testing
        trainset, testset = train_test_split(data, test_size=0.25)  # Example split

        # Define and train the model
        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        model.fit(trainset)

        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f'RMSE for n_factors={n_factors}, n_epochs={n_epochs}, lr_all={lr_all}, reg_all={reg_all}: {rmse}')
        
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_model_params = (n_factors, n_epochs, lr_all, reg_all)
            self.best_model = model

    def build_best_CF_model(self):
        n_factors_list = [50, 100, 150]
        n_epochs_list = [10, 20, 30]
        lr_all_list = [0.005, 0.01, 0.02]
        reg_all_list = [0.2, 0.4, 0.6]

        for n_factors in n_factors_list:
            for n_epochs in n_epochs_list:
                for lr_all in lr_all_list:
                    for reg_all in reg_all_list:
                        self.build_CF_model(n_factors, n_epochs, lr_all, reg_all)
        
        print("Best model parameters:", self.best_model_params)
        print("Best RMSE:", self.best_rmse)

    def get_movie_recommendations(self, user_id, n_recommendations=10):
        if not self.best_model:
            raise Exception("Model has not been trained. Call startup() first.")
        
        # Predict ratings for all movies the user hasn't rated yet and sort them
        predictions = []
        for movie_id in self.movies['movie_id'].unique():
            if not ((self.ratings['user_id'] == user_id) & (self.ratings['movie_id'] == movie_id)).any():
                predicted = self.best_model.predict(user_id, movie_id)
                predictions.append((movie_id, predicted.est))
                
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top N recommendations
        top_movie_ids = [pred[0] for pred in predictions[:n_recommendations]]
        top_movies = self.movies[self.movies['movie_id'].isin(top_movie_ids)]
        
        return top_movies[['movie_title', 'movie_genres']]
