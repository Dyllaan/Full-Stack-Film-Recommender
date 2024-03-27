import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from icp_api.models import Movie, Rating
import pandas as pd
from tensorflow.keras.models import load_model

class CFRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])

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

        try:
            self.load_model()
            print("Model loaded successfully.")
        except IOError:
            print("Model not found. Training a new model.")
            self.model = self.build_model()
            self.train_model()

    def build_model(self):
        user_input = Input(shape=(1,), name='user_input')
        user_emb = Embedding(output_dim=50, input_dim=self.n_users, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_emb)

        movie_input = Input(shape=(1,), name='movie_input')
        movie_emb = Embedding(output_dim=50, input_dim=self.n_movies, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='movie_flatten')(movie_emb)

        concat = Concatenate()([user_vec, movie_vec])
        dense = Dense(128, activation='relu')(concat)
        output = Dense(1)(dense)

        model = Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model
    
    def load_model(self):
        self.model = load_model('movie_recommender.h5')

    def train_model(self):
        self.model.fit([self.X_train[:, 0], self.X_train[:, 1]], self.y_train, batch_size=64, epochs=5, validation_data=([self.X_test[:, 0], self.X_test[:, 1]], self.y_test), verbose=1)
        self.model.save('movie_recommender.h5')

    def get_movie_recommendations(self, user_id, num_recommendations=10):
        
        if(user_id not in self.ratings['user_id'].values):
            return []
        
        user_id = self.user_enc.transform([user_id])[0]  # Transform the user ID

        movie_input = np.arange(self.n_movies)
        user_input = np.array([user_id] * self.n_movies)

        predictions = self.model.predict([user_input, movie_input])

        movie_ratings = list(zip(movie_input, predictions.flatten()))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)

        top_movie_recommendations = movie_ratings[:num_recommendations]
        top_movies = [{
            "movie_id": int(self.item_enc.inverse_transform([movie_id])[0]),
            "predicted_rating": float(rating),
            "movie_title": self.movies.loc[self.movies['movie_id'] == int(self.item_enc.inverse_transform([movie_id])[0])]['movie_title'].values[0]

        } for movie_id, rating in top_movie_recommendations]
        return top_movies