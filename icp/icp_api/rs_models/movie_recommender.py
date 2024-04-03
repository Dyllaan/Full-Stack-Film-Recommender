import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from math import sqrt
from keras.callbacks import EarlyStopping
from icp_api.models import Movie, Rating
from keras import regularizers
from surprise import SVD, Dataset, Reader
from bayes_opt import BayesianOptimization
import os
from keras.models import load_model

class CFRecommender:

    def startup(self):
        # Load data
        ratings_data = Rating.objects.all().values_list('user_id', 'movie_id', 'rating')
        movies_data = Movie.objects.all().values_list('movie_id', 'movie_title', 'movie_genres')
        
        self.ratings = pd.DataFrame(list(ratings_data), columns=['user_id', 'movie_id', 'rating'])
        self.movies = pd.DataFrame(list(movies_data), columns=['movie_id', 'movie_title', 'movie_genres'])
        
        found = self.load()
        print(os.getcwd())
        self.transform()

        if not found:
            self.optimize_hyperparameters()

    def load(self):
        try:
            self.model = load_model('best_model.keras')
            return True
        except:
            print("Model not found, please train the model first")
            return False

                
    def transform(self):
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

    def optimize_hyperparameters(self, init_points=5, n_iter=15):
        def objective(learning_rate, dropout_rate, batch_size, embedding_size):
            # Ensure the hyperparameters are within the method's scope
            self.learning_rate = learning_rate
            self.dropout_rate = dropout_rate
            self.batch_size = batch_size
            self.embedding_size = int(embedding_size)
            
            # Rebuild and retrain the model using these hyperparameters
            self.model = self.build_model()
            self.train_model(evaluate_only=True)  # Add an argument to train_model to return evaluation metrics without printing
            
            # For this example, let's assume lower MSE is better
            mse = self.model.evaluate([self.X_test[:, 0], self.X_test[:, 1]], self.y_test, verbose=0)[0]
            return -mse  # Maximize negative MSE because Bayesian Optimization seeks to maximize the objective function

        # Define the range for each hyperparameter
        pbounds = {
            'learning_rate': (1e-4, 1e-2),  # Range for learning rate
            'dropout_rate': (0.1, 0.5),     # Range for dropout rate
            'batch_size': (16, 128),        # Range for batch size
            'embedding_size': (16, 64)      # Range for embedding size
        }

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Optional: Print the best parameters found
        print("Best parameters found: ", optimizer.max['params'])

        # Use the best parameters to retrain the final model
        self.learning_rate = optimizer.max['params']['learning_rate']
        self.dropout_rate = optimizer.max['params']['dropout_rate']
        self.batch_size = optimizer.max['params']['batch_size']
        self.embedding_size = int(optimizer.max['params']['embedding_size'])
        self.model = self.build_model()
        self.train_model()
        #save best model
        self.model.save('best_model.keras')

    def build_model(self):
        #hidden_units = (32, 16, 8)
        user_input = Input(shape=(1,), name='user_input')
        user_emb = Embedding(output_dim=int(self.embedding_size), input_dim=self.n_users, name='user_embedding', embeddings_initializer="he_normal",
                                              embeddings_regularizer=regularizers.l1(1e-6))(user_input)
        user_vec = Flatten(name='user_flatten')(user_emb)
        user_vec = BatchNormalization()(user_vec)
        user_dense = Dense(int(self.embedding_size), activation='relu')(user_vec)
        #user_dense = LeakyReLU()(user_dense)
        user_dense = Dropout(self.dropout_rate)(user_dense)

        movie_input = Input(shape=(1,), name='movie_input')
        movie_emb = Embedding(output_dim=int(self.embedding_size), input_dim=self.n_movies, name='movie_embedding', embeddings_initializer="he_normal",
                                                embeddings_regularizer=regularizers.l1(1e-6))(movie_input)
        movie_vec = Flatten(name='movie_flatten')(movie_emb)
        movie_vec = BatchNormalization()(movie_vec)
        movie_dense = Dense(int(self.embedding_size), activation='relu')(movie_vec)
        #movie_dense = LeakyReLU()(movie_dense)
        movie_dense = Dropout(self.dropout_rate)(movie_dense)

        concat = Concatenate()([user_dense, movie_dense])
        out = Flatten()(concat)
        # Add one or more hidden layers
        #for n_hidden in hidden_units:
            #concat = Dense(n_hidden, activation='linear')(concat)
            #concat = LeakyReLU()(concat)  # Use LeakyReLU to add non-linearity
        out = Dropout(self.dropout_rate)(out)  # Add dropout layer
        out = Dense(1, activation='linear')(out)

        model = Model(inputs=[user_input, movie_input], outputs=out)
        model.compile(optimizer=Adam(self.learning_rate), loss='MSE', metrics=['MSE'])
        return model

    def train_model(self, evaluate_only=False):
        # Early stopping but only if the validation loss doesn't decrease by at least 0.001 for 5 epochs
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       min_delta=0.001, 
                                       patience=2, 
                                       restore_best_weights=True, 
                                       )  
        history = self.model.fit(
            [self.X_train[:, 0], self.X_train[:, 1]],  # Include genre data here
            self.y_train,
            batch_size=int(self.batch_size), 
            epochs=20, 
            validation_data=([self.X_test[:, 0], self.X_test[:, 1]], self.y_test), 
            verbose=1,
            callbacks=[early_stopping]
        )
        # Assuming model_evaluation is the result from model.evaluate() on your test set
        model_evaluation = self.model.evaluate([self.X_test[:, 0], self.X_test[:, 1]], self.y_test)

        if not evaluate_only:
            print(f"RMSE: {sqrt(model_evaluation[1])}")
        else:
            return model_evaluation

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
            "predicted_rating": rating
        } for movie_id, rating in top_movie_recommendations]

        return top_movies
