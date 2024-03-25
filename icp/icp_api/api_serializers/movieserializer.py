from rest_framework import serializers
from icp_api.models import Movie, TMDb, TMDbPoster
import requests

class MovieSerializer(serializers.ModelSerializer):
    poster_path = serializers.SerializerMethodField()

    class Meta:
        model = Movie
        fields = '__all__'

    def get_poster_path(self, obj):
        try:
            # Step 1: Find the TMDb record using the Movie's movie_id
            tmdb_record = TMDb.objects.get(movie_id=obj)
        
            # Step 2: Try to get the TMDbPoster record using the TMDb record's tmdb_id
            try:
                poster_record = TMDbPoster.objects.get(tmdb_id=tmdb_record.tmdb_id)
                if poster_record.poster_path:
                    return poster_record.poster_path
            except TMDbPoster.DoesNotExist:
                # If no poster record exists, proceed to fetch from the TMDb API
                pass

            # Step 3: Fetch from the TMDb API
            response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdb_record.tmdb_id}?api_key=YOUR_API_KEY_HERE')
            data = response.json()
            poster_path = data.get('poster_path')
        
            # Save the new poster path in TMDbPoster
            TMDbPoster.objects.create(tmdb_id=tmdb_record, poster_path=poster_path)  # Create a new TMDbPoster record
            return poster_path
        except TMDb.DoesNotExist:
            # Handle case where TMDb record does not exist for the given Movie's movie_id
            return None
