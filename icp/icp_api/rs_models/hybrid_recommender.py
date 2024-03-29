<<<<<<< Updated upstream
=======
import pandas as pd
from surprise import KNNBasic, SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from icp_api.models import Movie, Rating
from surprise.model_selection import cross_validate

class HybridRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])
        self.best_knn_params = None
        self.best_svd_params = None
        self.best_rmse = float('inf')
        self.best_knn_model = None
        self.best_svd_model = None
        self.build_best_models()

    def build_svd_model(self, n_factors, n_epochs, lr_all, reg_all):
        # Load the dataset from the DataFrame
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[['user_id', 'movie_id', 'rating']], reader)
        
        # Split the dataset into training and testing
        trainset, testset = train_test_split(data, test_size=0.25)  # Example split

        # Define and train the model
        model = SVD()
        cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
        model.fit(trainset)

        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f'SVD RMSE for n_factors={n_factors}, n_epochs={n_epochs}, lr_all={lr_all}, reg_all={reg_all}: {rmse}')
        
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_svd_params = (n_factors, n_epochs, lr_all, reg_all)
            self.best_svd_model = model

    def build_best_models(self):
        self.build_svd_model(50, 10, 0.005, 0.2)

    def get_movie_recommendations(self, user_id, n_recommendations=10):
        if not self.best_knn_model or not self.best_svd_model:
            raise Exception("Models have not been trained. Call startup() first.")
        
        # Predict ratings using both models and take the average
        predictions = []
        for movie_id in self.movies['movie_id'].unique():
            if not ((self.ratings['user_id'] == user_id) & (self.ratings['movie_id'] == movie_id)).any():
                knn_predicted = self.best_knn_model.predict(user_id, movie_id).est
                svd_predicted = self.best_svd_model.predict(user_id, movie_id).est
                ensemble_predicted = (knn_predicted + svd_predicted) / 2.0
                predictions.append((movie_id, ensemble_predicted))
                
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top N recommendations
        top_movie_ids = [pred[0] for pred in predictions[:n_recommendations]]
        top_movies = self.movies[self.movies['movie_id'].isin(top_movie_ids)]
        
        return top_movies[['movie_title', 'movie_genres']]
>>>>>>> Stashed changes
