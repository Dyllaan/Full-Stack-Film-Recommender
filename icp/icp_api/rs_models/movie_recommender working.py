import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from math import sqrt
from keras.callbacks import EarlyStopping
from icp_api.models import Movie, Rating
from sklearn.preprocessing import MultiLabelBinarizer

class CFRecommender:

    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])
        self.transform()

    def transform(self):
        self.ratings['rating'] = self.ratings['rating'].astype(float)

        #genres
        self.movies['movie_genres'] = self.movies['movie_genres'].apply(lambda x: x.split('|'))
        self.all_genres = sorted(set(genre for genres in self.movies['movie_genres'] for genre in genres))
        
        # Initialize the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit([self.all_genres])
        # Transform the genres into multi-hot encoded vectors
        self.movies['genres_encoded'] = list(mlb.transform(self.movies['movie_genres']))
        self.n_genres = len(self.all_genres)

        # Encoding user IDs and movie IDs
        self.user_enc = LabelEncoder()
        self.ratings['user'] = self.user_enc.fit_transform(self.ratings['user_id'].values)
        self.n_users = self.ratings['user'].nunique()

        self.item_enc = LabelEncoder()
        self.ratings['movie'] = self.item_enc.fit_transform(self.ratings['movie_id'].values)
        self.n_movies = self.ratings['movie'].nunique()

        # Convert genres_encoded from lists of encoded genres to a numpy array for easy concatenation
        self.genre_encoded = np.array(self.movies['genres_encoded'].tolist())

        # Splitting the dataset
        X = self.ratings[['user', 'movie']].values
        y = self.ratings['rating'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        self.model = self.build_model()
        self.train_model()

    def build_model(self):
        user_input = Input(shape=(1,), name='user_input')
        user_emb = Embedding(output_dim=75, input_dim=self.n_users, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_emb)

        movie_input = Input(shape=(1,), name='movie_input')
        movie_emb = Embedding(output_dim=75, input_dim=self.n_movies, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='movie_flatten')(movie_emb)

        # New input for genres
        genre_input = Input(shape=(self.n_genres,), name='genre_input')  # Shape matches number of genres
        genre_dense = Dense(32, activation='relu')(genre_input)

        # New: Combine movie vector with genre information first
        movie_genre_concat = Concatenate()([movie_vec, genre_dense])
        movie_genre_dense = Dense(64, activation='relu')(movie_genre_concat)

        concat = Concatenate()([user_vec, movie_genre_dense])
        dense = Dense(128, activation='relu')(concat)
        output = Dense(1)(dense)

        model = Model(inputs=[user_input, movie_input, genre_input], outputs=output)
        model.compile(optimizer=Adam(0.005), loss='mean_squared_error')
        return model

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)  
        self.model.fit(
            [self.X_train[:, 0], self.X_train[:, 1], self.genre_encoded[self.X_train[:, 1]]],  # Include genre data here
            self.y_train,
            batch_size=128, 
            epochs=20, 
            validation_data=([self.X_test[:, 0], self.X_test[:, 1], self.genre_encoded[self.X_test[:, 1]]], self.y_test), 
            verbose=1,
            callbacks=[early_stopping]
        )
        # Assuming model_evaluation is the result from model.evaluate() on your test set
        model_evaluation = self.model.evaluate([self.X_test[:, 0], self.X_test[:, 1], self.genre_encoded[self.X_test[:, 1]]], self.y_test)

        # model.evaluate() returns the mean squared error (MSE) since the model was compiled with loss='mean_squared_error'
        mse = model_evaluation

        # Calculate the root mean square error (RMSE)
        rmse = sqrt(mse)
        print(f"RMSE: {rmse}")

    def get_movie_recommendations(self, user_id_example, num_recommendations=10):
        user_id_example = self.user_enc.transform([user_id_example])[0]  # Transform the user ID
        movie_input = np.arange(self.n_movies)
        user_input = np.array([user_id_example] * self.n_movies)
        genre_input = self.genre_encoded[movie_input]

        predictions = self.model.predict([user_input, movie_input, genre_input])

        movie_ratings = list(zip(movie_input, predictions.flatten()))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)

        top_movie_recommendations = movie_ratings[:num_recommendations]
        top_movies = [{
            "movie_id": int(self.item_enc.inverse_transform([movie_id])[0]),
            "movie_title": self.movies[self.movies['movie_id'] == self.item_enc.inverse_transform([movie_id])[0]]['movie_title'].values[0],
            "predicted_rating": rating
        } for movie_id, rating in top_movie_recommendations]

        return top_movies
