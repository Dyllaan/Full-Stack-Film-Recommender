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
            tmdb_record = TMDb.objects.get(movie_id=obj.movie_id)  # Ensure you're using obj.id to get the movie_id correctly
            
            # Step 2: Try to get the TMDbPoster record using the TMDb record's tmdb_id
            poster_record = TMDbPoster.objects.filter(tmdb_id=tmdb_record.tmdb_id).first()
            if poster_record and poster_record.poster_path:
                return poster_record.poster_path

            # Step 3: Fetch from the TMDb API if no poster record exists or it doesn't have a poster_path
            response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdb_record.tmdb_id}?api_key=bb74221ae90248cbc87b1360be4ee33e')
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                
                if poster_path:  # Check if poster_path is not None or empty
                    # Save the new poster path in TMDbPoster if a poster path is found
                    TMDbPoster.objects.create(tmdb_id=tmdb_record, poster_path=poster_path)
                    return poster_path

        except TMDb.DoesNotExist:
            # Handle case where TMDb record does not exist for the given Movie's movie_id
            return None
        except TMDbPoster.DoesNotExist:
            # This catch is no longer necessary since we use filter().first() now, but kept for reference
            pass

        # Return None if no poster is found or any other exception occurs
        return None
