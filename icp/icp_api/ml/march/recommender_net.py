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
from gensim.models import Word2Vec

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

    def build(self, input_shape):
        super(RecommenderNet, self).build(input_shape)

    def call(self, inputs):
        user_input, movie_input = inputs[:, 0], inputs[:, 1]
        genre_inputs = inputs[:, 2:]

        
        user_vector = self.user_embedding(user_input)
        user_bias = self.user_bias(user_input)
        
        movie_vector = self.movie_embedding(movie_input)
        movie_vector = self.movie_dropout(movie_vector)  # Apply dropout to movie vector
        movie_bias = self.movie_bias(movie_input)

        # Process genre inputs
        genre_vector = self.genre_dense(genre_inputs)
        

        # Combine all features
        combined_features = tf.concat([user_vector, movie_vector, genre_vector], axis=1)
        
        # Process combined features with dense layers
        x = self.combined_dense1(combined_features)
        x = self.combined_dense2(x)
        
        # Add biases
        x = x + user_bias + movie_bias
        
        return tf.nn.sigmoid(x)
