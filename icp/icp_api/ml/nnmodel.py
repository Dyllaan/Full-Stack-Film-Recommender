import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np


# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# For simplicity, let's use only ratings for collaborative filtering
# Encoding user IDs and movie IDs
user_enc = LabelEncoder()
ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
n_users = ratings['user'].nunique()

item_enc = LabelEncoder()
ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)
n_movies = ratings['movie'].nunique()

# Train-test split
X = ratings[['user', 'movie']].values
y = ratings['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Neural collaborative filtering model
user_input = Input(shape=(1,), name='user_input')
user_emb = Embedding(output_dim=50, input_dim=n_users, input_length=1, name='user_embedding')(user_input)
user_vec = Flatten(name='user_flatten')(user_emb)

movie_input = Input(shape=(1,), name='movie_input')
movie_emb = Embedding(output_dim=50, input_dim=n_movies, input_length=1, name='movie_embedding')(movie_input)
movie_vec = Flatten(name='movie_flatten')(movie_emb)

concat = Concatenate()([user_vec, movie_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1)(dense)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

# Train the model
model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=64, epochs=5, validation_data=([X_test[:, 0], X_test[:, 1]], y_test), verbose=1)
# Evaluate the model on the test data (RMSE)
rmse = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print("Root Mean Squared Error (RMSE):", rmse)

# Make movie recommendations for a user
user_id_example = 1  # Replace with the desired user ID
user_movies = ratings[ratings['user'] == user_id_example]['movie'].unique()
unrated_movies = [movie for movie in range(n_movies) if movie not in user_movies]

# Predict ratings for unrated movies
user_input = [user_id_example] * len(unrated_movies)
# Create an array of user_input and unrated_movies as input tensors
user_input_tensor = np.array(user_input)
unrated_movies_tensor = np.array(unrated_movies)
predictions = model.predict([user_input_tensor, unrated_movies_tensor])


# Combine movie IDs and predictions
movie_predictions = list(zip(unrated_movies, predictions))
# Sort by predicted ratings in descending order
movie_predictions.sort(key=lambda x: x[1], reverse=True)
# Get top N recommendations (e.g., top 10)
top_recommendations = [item_enc.inverse_transform([x[0]])[0] for x in movie_predictions[:10]]
print("Top Recommendations for User", user_id_example, ":", top_recommendations)

top_recommendations_with_titles = [
    (item_enc.inverse_transform([x[0]])[0], movies[movies['movieId'] == item_enc.inverse_transform([x[0]])[0]]['title'].values[0])
    for x in movie_predictions[:10]
]

print("Top Recommendations for User", user_id_example, ":")
for movie_id, movie_title in top_recommendations_with_titles:
    print(f"Movie ID: {movie_id}, Title: {movie_title}")