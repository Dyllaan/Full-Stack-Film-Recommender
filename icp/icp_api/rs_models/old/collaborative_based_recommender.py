import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from icp_api.models import Movie, Rating
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU


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

        # Prepare genres data
        self.prepare_genres()

        # Splitting the dataset
        X_user_movie = self.ratings[['user', 'movie']].values
        X_genre = self.movie_genres_binarized[self.ratings['movie'].values]
        y = self.ratings['rating'].values
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_genre_train, self.X_genre_test = train_test_split(X_user_movie, y, X_genre, test_size=0.1, random_state=42)
       
        self.model = self.build_model()
        self.train_model()

    def prepare_genres(self):
        # Assuming 'movie_genres' is a '|' separated string of genres
        self.movies['genres_list'] = self.movies['movie_genres'].apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        self.movie_genres_binarized = mlb.fit_transform(self.movies['genres_list'])
        self.n_genres = self.movie_genres_binarized.shape[1]

    def build_model(self):
        user_input = Input(shape=(1,), name='user_input')
        user_emb = Embedding(output_dim=50, input_dim=self.n_users, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_emb)

        movie_input = Input(shape=(1,), name='movie_input')
        movie_emb = Embedding(output_dim=50, input_dim=self.n_movies, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='movie_flatten')(movie_emb)

        genre_input = Input(shape=(self.n_genres,), name='genre_input')
        
        concat = Concatenate()([user_vec, movie_vec, genre_input])

        # Apply Batch Normalization before activation
        dense = Dense(128)(concat)
        batch_norm = BatchNormalization()(dense)
        leaky_relu = LeakyReLU()(batch_norm)

        # Add Dropout for regularization
        dropout = Dropout(0.5)(leaky_relu)
        
        output = Dense(1)(dropout)

        model = Model(inputs=[user_input, movie_input, genre_input], outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model
    
    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', restore_best_weights=True)
        self.model.fit(
            [self.X_train[:, 0], self.X_train[:, 1], self.X_genre_train], self.y_train,
            batch_size=64,
            epochs=100,
            validation_data=([self.X_test[:, 0], self.X_test[:, 1], self.X_genre_test], self.y_test),
            verbose=1,
            callbacks=[early_stopping]
        )
        self.calculate_rmse()

    def calculate_rmse(self):
        # Assuming self.X_test is [users, movies] and self.X_genre_test contains the genre data for the test set
        predictions = self.model.predict([self.X_test[:, 0], self.X_test[:, 1], self.X_genre_test])
        mse = np.mean(np.square(predictions - self.y_test))
        rmse = np.sqrt(mse)
        print(f'Test RMSE: {rmse:.4f}')
