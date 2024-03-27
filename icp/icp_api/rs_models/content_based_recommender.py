import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from icp_api.models import Movie, Rating, Tag
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model

class CBFRecommender:
    
    def startup(self):
        # Load data
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres', 'release_year')
        tags_data = Tag.objects.all().values_list('movie_id', 'tag')

        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres', 'release_year'])
        self.tags = pd.DataFrame(list(tags_data), columns=['movie_id', 'tag'])
        
        # Prepare movie features and compute cosine similarity after loading and processing the data
        self.prepare_movie_features()
        self.compute_cosine_similarity()

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

        # Convert release year to string and include it in combined features
        self.movies['release_year_str'] = self.movies['release_year'].astype(str)
        self.movies['combined_features'] = self.movies['movie_title'] + ' ' + \
                                    self.movies['genres_str'] + ' ' + \
                                    self.movies['tags_str'] + ' ' + \
                                    self.movies['release_year_str']



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
    
    
    def get_recommendations_for_user(self, user_movie_titles, num_recommendations=10):
        # Find indices for the user's movies
        indices = self.movies[self.movies['movie_title'].isin(user_movie_titles)].index.tolist()
        if not indices:
            return []  # Return an empty list if no movies are found

        # Compute the average feature vector for the user's movies
        user_vector = np.mean(self.feature_matrix[indices].toarray(), axis=0)

        # Compute cosine similarity between user vector and all movie features
        cosine_sim = cosine_similarity(user_vector.reshape(1, -1), self.feature_matrix).flatten()

        # Get the scores of the most similar movies
        sim_scores = list(enumerate(cosine_sim))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top N most similar movies
        sim_scores = sim_scores[:num_recommendations]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top N most similar titles along with their IDs
        return [{"movie_id": int(self.movies.iloc[idx]['movie_id']), "movie_title": self.movies.iloc[idx]['movie_title']} for idx in movie_indices]
