import requests
import csv
import pandas as pd

# Replace 'YOUR_API_KEY' with your actual TMDb API key
api_key = 'bb74221ae90248cbc87b1360be4ee33e'
api_base_url = 'https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US'

# Function to get poster path for a given TMDB ID
def get_poster_path(tmdb_id):
    try:
        response = requests.get(api_base_url.format(tmdb_id, api_key))
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()
        return data.get('poster_path')
    except requests.RequestException:
        return None

# Read TMDB IDs from CSV
tmdb_ids_df = pd.read_csv('links.csv')  # Assuming the TMDB ID column is named 'tmdbId'

# Prepare a list to store TMDB IDs and their corresponding poster paths
movie_posters = []

# Retrieve poster path for each TMDB ID and append to the list
for tmdb_id in tmdb_ids_df['tmdbId']:
    poster_path = get_poster_path(tmdb_id)
    if poster_path:
        movie_posters.append({'tmdbId': tmdb_id, 'posterPath': poster_path})

# Convert the list to a DataFrame and save to a new CSV file
posters_df = pd.DataFrame(movie_posters)
posters_df.to_csv('movie_posters.csv', index=False)

print("Done! Check the movie_posters.csv file for the poster paths.")
