import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from recommender_net import RecommenderNet

class CFRecommender:

    EMBEDDING_SIZE = 50

    def load_data(self):
        ratings_file = r"C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\ratings.csv"
        self.ratings_df = pd.read_csv(ratings_file)

        moviesFile = r"C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\movies.csv"
        self.movie_df = pd.read_csv(moviesFile)

        tagsFile = r"C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\tags.csv"
        self.tags_df = pd.read_csv(tagsFile)

        self.prepare_data()

    def prepare_word2vec(self):
        # Assuming self.tags_df is already loaded with userId, movieId, tag, and timestamp columns
        # Group tags by movieId and aggregate them into a list
        self.tags_df['processed_tags'] = self.tags_df['tag'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
        movie_tags = self.tags_df.groupby('movieId')['processed_tags'].apply(list).reset_index()
        model = Word2Vec(sentences=movie_tags['processed_tags'], vector_size=50, window=5, min_count=1, workers=4)

        # Save the model for later use
        model.save("movie_tags_word2vec.model")

        movie_tags['tag_embedding'] = movie_tags['processed_tags'].apply(lambda x: self.get_movie_tag_embedding(x, model))

        # Ensure the tag embeddings are in the correct format for model training
        return np.stack(movie_tags['tag_embedding'].values)

    def get_movie_tag_embedding(self, movie_tags_row, w2v_model):
        # Average the Word2Vec vectors for all tags associated with the movie
        vectors = [w2v_model.wv[tag] for tag in movie_tags_row if tag in w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(w2v_model.vector_size)

    
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
        

        # Convert genres_encoded from lists of encoded genres to a numpy array for easy concatenation
        genre_encoded = np.array(self.movie_df['genres_encoded'].tolist())
        
        # You now need to ensure that the user and movie inputs are combined with the genre information
        x = np.hstack([
            self.ratings_df[['user', 'movie']].values, 
            genre_encoded[self.ratings_df['movie'].values]  # Lookup the genre encoding based on the movie index
        ])
        
        # Your existing code to split into training and validation sets follows
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
        self.model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.005))

        # Setup EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, restore_best_weights=True)

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=2,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping]
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

        movies_not_watched_indices = [self.movie2movie_encoded.get(x) for x in movies_not_watched]
        user_encoder = self.user2user_encoded.get(user_id)
        
        # Prepare the genre data for movies not watched by user
        # Ensure the genre_encoded array is properly aligned with the movie indices
        genres_not_watched = np.array(self.movie_df['genres_encoded'].tolist())[movies_not_watched_indices]
        
        # Create an array with repeated user index for length of movies_not_watched list
        user_indices = np.array([user_encoder] * len(movies_not_watched_indices))
        
        # Combine user indices, movie indices, and genres into a single array for prediction
        user_movie_genre_array = np.hstack((user_indices[:, None], np.array(movies_not_watched_indices)[:, None], genres_not_watched))

        # Predict the ratings for the not watched movies with genres
        ratings = self.model.predict(user_movie_genre_array).flatten()

        # Get the top 10 ratings indices
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_movie_ids = [
            self.movie_encoded2movie.get(movies_not_watched_indices[x]) for x in top_ratings_indices
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

recommender = CFRecommender()
recommender.run()