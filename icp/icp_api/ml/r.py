import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.2)
model.fit(trainset)

test_predictions = model.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(test_predictions)
print(f'RMSE: {rmse}')

# Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(movie_title):
    idx = movies.index[movies['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

def hybrid_recommendations(userId, movie_title):
    idx = movies.index[movies['title'] == movie_title].tolist()[0]
    movieId = movies['movieId'][idx]
    
    # Get collaborative filtering predictions
    cf_predictions = []
    for m_id in movies['movieId']:
        prediction = model.predict(userId, m_id)
        cf_predictions.append((m_id, prediction.est))
    
    # Sort predictions
    cf_predictions = sorted(cf_predictions, key=lambda x: x[1], reverse=True)
    
    # Get content-based recommendations
    cb_recommendations = get_content_based_recommendations(movie_title)
    
    # Merge the two recommendation sets
    hybrid_recommendations = []
    for rec in cb_recommendations:
        m_id = movies['movieId'][movies['title'] == rec].tolist()[0]
        score = next((pred[1] for pred in cf_predictions if pred[0] == m_id), None)
        hybrid_recommendations.append((rec, score))
    
    # Sort by collaborative filtering score
    hybrid_recommendations = sorted(hybrid_recommendations, key=lambda x: x[1], reverse=True)
    
    return hybrid_recommendations

# Get hybrid recommendations for a user and a movie
recommendations = hybrid_recommendations(1, 'Toy Story (1995)')
for rec in recommendations:
    print(f'Movie: {rec[0]}, Score: {rec[1]}')
