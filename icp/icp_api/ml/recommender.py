import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

# Load the data from ratings.csv
data = pd.read_csv('ratings.csv')

# The columns must correspond to user id, item id and ratings (in that order).
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use user-based collaborative filtering
sim_options = {
    'name': 'cosine',
    'user_based': True  # compute similarities between users
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Make predictions and evaluate the model
predictions = model.test(testset)
accuracy.rmse(predictions)
