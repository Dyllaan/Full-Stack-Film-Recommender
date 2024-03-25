import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from icp_api.models import Movie, Rating
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

class MovieRecommender:
    
    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', flat=True)
        tags_data = Tag.objects.all().values_list('movie_id', 'tag')

        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id'])

        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data))
        self.tags = pd.DataFrame(list(tags_data), columns=['movie_id', 'tag'])

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

    def train_model(self):
        self.model.fit([self.X_train[:, 0], self.X_train[:, 1]], self.y_train, batch_size=64, epochs=5, validation_data=([self.X_test[:, 0], self.X_test[:, 1]], self.y_test), verbose=1)

    def get_movie_recommendations(self, user_id_example, num_recommendations=10):
        user_id_example = self.user_enc.transform([user_id_example])[0]  # Transform the user ID
        movie_input = np.arange(self.n_movies)
        user_input = np.array([user_id_example] * self.n_movies)

        predictions = self.model.predict([user_input, movie_input])

        movie_ratings = list(zip(movie_input, predictions.flatten()))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)

        top_movie_recommendations = movie_ratings[:num_recommendations]
        top_movies = [{
            "movie_id": int(self.item_enc.inverse_transform([movie_id])[0]),
            "predicted_rating": float(rating)  # Cast to float here
        } for movie_id, rating in top_movie_recommendations]
        return top_movies

    def prepare_movie_features(self):
        # Process genres
        self.movies['movie_genres'] = self.movies['movie_genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        self.movies['genres_str'] = self.movies['movie_genres'].apply(lambda x: ' '.join(x))

        # Process tags by combining them into a single string per movie
        combined_tags = self.tags.groupby('movie_id')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        combined_tags.rename(columns={'tag': 'tags_str'}, inplace=True)

        # Merge movies with their combined tags
        self.movies = pd.merge(self.movies, combined_tags, on='movie_id', how='left')
        self.movies['tags_str'] = self.movies['tags_str'].fillna('')  # Fill missing tags with empty string

        # Combine genres and tags into a single feature
        self.movies['combined_features'] = self.movies['genres_str'] + ' ' + self.movies['tags_str']

        # Vectorize the combined features using TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(self.movies['combined_features'])

    def compute_cosine_similarity(self):
        # Compute the cosine similarity matrix from the feature matrix
        self.cosine_sim = cosine_similarity(self.feature_matrix, self.feature_matrix)

    def get_content_based_recommendations(self, movie_title, num_recommendations=10):
        # Make sure to call compute_cosine_similarity somewhere before this function to prepare self.cosine_sim
        # Find the index of the movie with the given title
        idx = self.movies.index[self.movies['movie_title'] == movie_title].tolist()
        if not idx:
            return []  # If the movie isn't found, return an empty list
        idx = idx[0]

        # Get pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:num_recommendations+1]  # Skip the first one because it's the movie itself

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar titles
        return self.movies['movie_title'].iloc[movie_indices].tolist()