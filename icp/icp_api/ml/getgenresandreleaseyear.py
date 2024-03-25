import pandas as pd
from io import StringIO

# Read the data into a DataFrame
df = pd.read_csv("movies.csv")

# Step 2: Extract the year from the title and remove it from the title string
df['release_year'] = df['title'].str.extract(r'\((\d{4})\)').astype(int)
df['title'] = df['title'].str.replace(r' \(\d{4}\)', '', regex=True)

# Step 3: Split the genres into individual ones and prepare data for a new "genres.csv" file
genres_data = df['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='genre')
genres_data.rename(columns={'index': 'movieId'}, inplace=True)

# Step 4: Create a new DataFrame for the modified movies data
movies_new = df[['movieId', 'title', 'release_year']]

# Step 5: Create a separate DataFrame for the genres data
genres_csv = genres_data[['movieId', 'genre']]

# Step 6: Save both DataFrames to new CSV files
movies_new.to_csv('/mnt/data/movies_new.csv', index=False)
genres_csv.to_csv('/mnt/data/genres.csv', index=False)

# Output file paths for download
movies_new_path = 'movies_new.csv'
genres_csv_path = '/genres_new.csv'

movies_new_path, genres_csv_path
