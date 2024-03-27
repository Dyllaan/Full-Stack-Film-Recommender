- Copy Movies
- \copy movies(movie_id, movie_title, movie_genres) FROM 'C:\Users\Louis\Desktop\ICP\ml-latest-small\utf8movies.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');


- Copy Links
- \copy links(movie_id, imdb_id, tmdb_id) FROM 'C:\Users\Louis\Desktop\Project code rework\ml-latest-small\links.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

- Copy Posters
- \copy movies(tmdb_id, poster_path) FROM 'C:\Users\Louis\Desktop\ICP\ml-latest-small\movie_posters.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');


- Copy Posters and links
- \copy posters(tmdb_id, poster_path) FROM 'C:\Users\Louis\Desktop\ICP\ml-latest-small\movie_posters.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

- Copy TMDb data
- \copy tmdb(movietmdb_id) FROM 'C:\Users\Louis\Desktop\ICP\ml-latest-small\movie_posters.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

- Copy Ratings data
- \copy icp_api_rating(user_id, movie_id, rating, created_at) FROM 'C:\Users\Louis\Desktop\ICP\ml-latest-small\ratings.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

- Copy Tags data
- \copy icp_api_tag(user_id, movie_id, tag, timestamp) FROM 'C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\ratings.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');



IGNORE DUPLICATES

INSERT INTO tmdb (movie_id, tmdb_id)
SELECT movie_id, tmdb_id
FROM links
WHERE tmdb_id IS NOT NULL
ON CONFLICT (tmdb_id) DO NOTHING;

- Remove year and set it to the release year column
UPDATE movies
SET movie_title = REGEXP_REPLACE(movie_title, '\s*\(\d{4}\)$', '', 'g');

