import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge movies with ratings
ratings = pd.merge(ratings, movies[['movieId', 'genres']], on='movieId', how='left')

# User encoding
user_enc = LabelEncoder()
ratings['user'] = user_enc.fit_transform(ratings['userId'].values)

# Item encoding
item_enc = LabelEncoder()
ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)

# Genre encoding (one-hot encoding)
genre_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Adjust as needed
genres_onehot = genre_enc.fit_transform(ratings[['genres']].values.reshape(-1, 1))

# Normalize features
timestamp_scaler = MinMaxScaler()
ratings['timestamp'] = timestamp_scaler.fit_transform(ratings[['timestamp']])

# Prepare inputs
X = ratings[['user', 'movie', 'timestamp']].join(pd.DataFrame(genres_onehot)).values
y = ratings['rating'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model architecture
user_input = Input(shape=(1,), name='user_input')
user_emb = Embedding(output_dim=50, input_dim=ratings['user'].nunique(), input_length=1, name='user_embedding', embeddings_regularizer=l2(0.001))(user_input)
user_vec = Flatten(name='user_flatten')(user_emb)

movie_input = Input(shape=(1,), name='movie_input')
movie_emb = Embedding(output_dim=50, input_dim=ratings['movie'].nunique(), input_length=1, name='movie_embedding', embeddings_regularizer=l2(0.001))(movie_input)
movie_vec = Flatten(name='movie_flatten')(movie_emb)

timestamp_input = Input(shape=(1,), name='timestamp_input')

genre_input = Input(shape=(genres_onehot.shape[1],), name='genre_input')
genre_emb = Embedding(input_dim=genres_onehot.shape[1], output_dim=32, name='genre_embedding')(genre_input)  # Adjust output_dim as needed
genre_vec = Flatten(name='genre_flatten')(genre_emb)

concat = Concatenate()([user_vec, movie_vec, timestamp_input, genre_vec])
dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concat)
dense = Dropout(0.5)(dense)  # Dropout for regularization
output = Dense(1)(dense)

model = Model(inputs=[user_input, movie_input, timestamp_input, genre_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit([X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3:]], y_train, batch_size=64, epochs=100, validation_data=([X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3:]], y_test), callbacks=[early_stopping], verbose=1)

# Evaluate the model
mse = model.evaluate([X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3:]], y_test)

print("Test RMSE:", sqrt(mse))

# Example: Get top 10 movie recommendations for a specific user (replace with desired user ID)
# This function remains as an example, actual recommendation functionality would need more than one entry to work properly.
def get_movie_recommendations(user_id_example, num_recommendations=10):
    movie_input = np.arange(len(movies))  # Create an array of all movie IDs
    user_input = np.array([user_id_example] * len(movies))  # Repeat the user ID for all movies
    timestamp_input = np.zeros(len(movies))  # Dummy timestamps

    # Predict ratings for the given user and all movies
    predictions = model.predict([user_input, movie_input, timestamp_input])

    # Combine movie IDs and predictions
    movie_ratings = list(zip(movie_input, predictions.flatten()))

    # Sort by predicted ratings in descending order
    movie_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get top N movie recommendations
    top_movie_recommendations = [x[0] for x in movie_ratings[:num_recommendations]]

    return top_movie_recommendations

user_id_example = 0  # User ID example (after encoding)
top_movie_recommendations = get_movie_recommendations(user_id_example, num_recommendations=10)

print("Top Movie Recommendations for User", user_enc.inverse_transform([user_id_example])[0], ":")
for movie_id in top_movie_recommendations:
    print("Movie ID:", movie_id, "Movie Title:", movies[movies['movieId'] == item_enc.inverse_transform([movie_id])[0]]['title'].values[0])
