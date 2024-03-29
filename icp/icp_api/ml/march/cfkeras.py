import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras import ops
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from keras import regularizers

class CFRecommender:

    EMBEDDING_SIZE = 50

    def load_data(self):
        ratings_file = r"C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\ratings.csv"
        self.ratings_df = pd.read_csv(ratings_file)

        moviesFile = r"C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\movies.csv"
        self.movie_df = pd.read_csv(moviesFile)

        self.prepare_data()
    
    
    def prepare_data(self):
        #genres
        self.movie_df['genres'] = self.movie_df['genres'].apply(lambda x: x.split('|'))
        self.all_genres = sorted(set(genre for genres in self.movie_df['genres'] for genre in genres))
        # Initialize the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit([self.all_genres])
        # Transform the genres into multi-hot encoded vectors
        self.movie_df['genres_encoded'] = list(mlb.transform(self.movie_df['genres']))
        # Check the shape of the encoded genres to confirm size (num_movies, num_genres)
        print("Encoded genres shape:", len(self.movie_df['genres_encoded'][0]))

        user_ids = self.ratings_df["userId"].unique().tolist()
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}
        movie_ids = self.ratings_df["movieId"].unique().tolist()
        self.movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        self.ratings_df["user"] = self.ratings_df["userId"].map(self.user2user_encoded)
        self.ratings_df["movie"] = self.ratings_df["movieId"].map(self.movie2movie_encoded)
        num_users = len(self.user2user_encoded)
        num_movies = len(self.movie_encoded2movie)
        num_genres = len(self.all_genres)
        self.ratings_df["rating"] = self.ratings_df["rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        self.min_rating = min(self.ratings_df["rating"])
        self.max_rating = max(self.ratings_df["rating"])

        self.ratings_df = self.ratings_df.sample(frac=1, random_state=42)
        x = self.ratings_df[["user", "movie"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = self.ratings_df["rating"].apply(lambda x: (x - self.min_rating) / (self.max_rating - self.min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * self.ratings_df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )
        self.build_model(num_users, num_movies, num_genres, x_train, x_val, y_train, y_val)

    def build_model(self, num_users, num_movies, num_genres, x_train, x_val, y_train, y_val):
        # Now modify the self.model instantiation
        self.model = RecommenderNet(num_users, num_movies, num_genres, self.EMBEDDING_SIZE, dropout_rate=0.5)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001))

        # Setup EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, restore_best_weights=True)

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=3,
            verbose=1,
            validation_data=(x_val, y_val),
        )
        self.evaluate_model(x_val, y_val, self.min_rating, self.max_rating, self.history)
    
    def evaluate_model(self, x_val, y_val, min_rating, max_rating, history):
        # Predict the normalized ratings for the validation set
        y_pred_norm = self.model.predict(x_val).flatten()

        # Rescale the predictions and actual ratings back to the original rating scale
        y_pred = y_pred_norm * (max_rating - min_rating) + min_rating
        y_true = y_val * (max_rating - min_rating) + min_rating

        # Calculate RMSE
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print(f'RMSE: {rmse}')

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    
    def recommend(self):
        # Let us get a user and see the top recommendations.
        user_id = self.ratings_df.userId.sample(1).iloc[0]
        movies_watched_by_user = self.ratings_df[self.ratings_df.userId == user_id]
        movies_not_watched = self.movie_df[
            ~self.movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
        ]["movieId"]
        movies_not_watched = list( 
            set(movies_not_watched).intersection(set(self.movie2movie_encoded.keys()))
        )
        movies_not_watched = [[self.movie2movie_encoded.get(x)] for x in movies_not_watched]
        user_encoder = self.user2user_encoded.get(user_id)
        user_movie_array = np.hstack(
            ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
        )
        ratings = self.model.predict(user_movie_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_movie_ids = [
            self.movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
        ]

        print("Showing recommendations for user: {}".format(user_id))
        print("====" * 9)
        print("Movies with high ratings from user")
        print("----" * 8)
        top_movies_user = (
            movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
        )
        movie_df_rows = self.movie_df[self.movie_df["movieId"].isin(top_movies_user)]
        for row in movie_df_rows.itertuples():
            print(row.title, ":", row.genres)

        print("----" * 8)
        print("Top 10 movie recommendations")
        print("----" * 8)
        recommended_movies = self.movie_df[self.movie_df["movieId"].isin(recommended_movie_ids)]
        for row in recommended_movies.itertuples():
            print(row.title, ":", row.genres)

    def save_model(self):
        self.model.save('trained_model.keras')
    
    def run(self):
        self.load_data()
        self.save_model()
        self.recommend()

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, num_genres, embedding_size, dropout_rate=0.5, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.embedding_size = embedding_size
        
        # User and Movie Embeddings
        self.user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size,
                                               embeddings_initializer="he_normal",
                                               embeddings_regularizer=regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)
        
        self.movie_embedding = layers.Embedding(input_dim=num_movies, output_dim=embedding_size,
                                                embeddings_initializer="he_normal",
                                                embeddings_regularizer=regularizers.l2(1e-6))
        self.movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)
        
        # Dropout layer for movie embedding
        self.movie_dropout = layers.Dropout(dropout_rate)

        # Dense layer for processing genres - assuming direct input of encoded genres as dense vector
        self.genre_dense = layers.Dense(embedding_size, activation="relu")

        # Additional Dense layers for combining features
        self.combined_dense1 = layers.Dense(64, activation="relu")
        self.combined_dense2 = layers.Dense(1)

    def call(self, inputs):
        # Assuming inputs = [user_input, movie_input], without genre input for simplicity
        user_input, movie_input = inputs[:, 0], inputs[:, 1]
        
        user_vector = self.user_embedding(user_input)
        user_bias = self.user_bias(user_input)
        
        movie_vector = self.movie_embedding(movie_input)
        movie_vector = self.movie_dropout(movie_vector)  # Apply dropout to movie vector
        movie_bias = self.movie_bias(movie_input)
        
        # Combine features: user_vector, movie_vector (genre_vector is optional, depends on your input)
        combined_features = tf.concat([user_vector, movie_vector], axis=1)
        
        # Process combined features with dense layers
        x = self.combined_dense1(combined_features)
        x = self.combined_dense2(x)
        
        # Add biases
        x = x + user_bias + movie_bias
        
        return tf.nn.sigmoid(x)

    
recommender = CFRecommender()
recommender.run()