import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, Embedding, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from icp_api.models import Movie, Rating
from tensorflow_recommenders import layers, models
from background_task import background
from tensorflow.keras.callbacks import EarlyStopping

class CFRecommenderModel(tf.keras.Model):
    def __init__(self, user_dim, movie_dim, embedding_dim=50):
        super().__init__()
        self.user_embedding = Embedding(user_dim, embedding_dim)
        self.movie_embedding = Embedding(movie_dim, embedding_dim)
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs["user"])
        user_vector = Flatten()(user_vector)
        
        movie_vector = self.movie_embedding(inputs["movie"])
        movie_vector = Flatten()(movie_vector)
        
        concat = tf.concat([user_vector, movie_vector], axis=1)
        dense_output = self.dense1(concat)
        return self.dense2(dense_output)

class CFRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')

        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])

        self.ratings['rating'] = self.ratings['rating'].astype(float)

        # Encoding user IDs and movie IDs
        self.user_enc = LabelEncoder()
        self.ratings['user'] = self.user_enc.fit_transform(self.ratings['user_id'].values)
        self.n_users = self.ratings['user'].nunique()

        self.item_enc = LabelEncoder()
        self.ratings['movie'] = self.item_enc.fit_transform(self.ratings['movie_id'].values)
        self.n_movies = self.ratings['movie'].nunique()

        # Splitting the dataset
        X = self.ratings[['user', 'movie']].values
        y = self.ratings['rating'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        self.model = CFRecommenderModel(self.n_users, self.n_movies)
        self.model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        self.train_model()

    @background(schedule=180)
    def update_and_retrain(self):
        # This function can remain largely the same, with adjustments for model loading/saving if needed.
        pass

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', restore_best_weights=True)

        self.model.fit(
            {"user": self.X_train[:, 0], "movie": self.X_train[:, 1]},
            self.y_train,
            batch_size=64,
            epochs=100,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stopping]  # Add the EarlyStopping callback here
        )

        # Evaluate the model on the test set after training
        self.evaluate_model(self.X_test, self.y_test)


    def evaluate_model(self, X, y_true):
        """
        Evaluate the model on the given dataset and print the RMSE.
        
        :param X: A 2D array of inputs, where the first column is user IDs, and the second column is movie IDs.
        :param y_true: The true ratings corresponding to the rows in X.
        """
        user_input = X[:, 0]
        movie_input = X[:, 1]
        
        predictions = self.model.predict({'user': user_input, 'movie': movie_input})
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((y_true - predictions.flatten())**2))
        print("RMSE:", rmse)

    

    def get_movie_recommendations(self, user_id, num_recommendations=10):
        
        if(user_id not in self.ratings['user_id'].values):
            return []
        
        user_id = self.user_enc.transform([user_id])[0]  # Transform the user ID

        movie_input = np.arange(self.n_movies)
        user_input = np.array([user_id] * self.n_movies)

        #predictions = self.model.predict([user_input, movie_input])
        predictions = self.model.predict({'user': user_input, 'movie': movie_input})

        movie_ratings = list(zip(movie_input, predictions.flatten()))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)

        top_movie_recommendations = movie_ratings[:num_recommendations]
        top_movies = [{
            "movie_id": int(self.item_enc.inverse_transform([movie_id])[0]),
            "predicted_rating": float(rating),
            "movie_title": self.movies.loc[self.movies['movie_id'] == int(self.item_enc.inverse_transform([movie_id])[0])]['movie_title'].values[0]

        } for movie_id, rating in top_movie_recommendations]
        return top_movies