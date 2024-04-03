from rest_framework import serializers
from icp_api.models import Movie, TMDb, TMDbPoster, MovieSlug
import requests

class MovieSerializer(serializers.ModelSerializer):
    poster_path = serializers.SerializerMethodField()
    movie_slug = serializers.SerializerMethodField()  # Add the new field to the serializer
    tmdb_id = serializers.SerializerMethodField()
    

    
    class Meta:
        model = Movie
        exclude = ['movie_id']

    #https://stackoverflow.com/questions/31820389/can-to-representation-in-django-rest-framework-access-the-normal-fields
    def to_representation(self, instance):
        ret = super().to_representation(instance)
        
        prediction_rating = self.context.get('prediction_ratings', {}).get(str(instance.movie_id), None)
        if prediction_rating is not None:
            ret['prediction_rating'] = prediction_rating
        return ret

    def get_poster_path(self, obj):
        try:
            # Find the TMDb record using the Movie's movie_id
            tmdb_record = TMDb.objects.get(movie_id=obj.movie_id)  # Ensure you're using obj.id to get the movie_id correctly
            
            # Try to get the TMDbPoster record using the TMDb record's tmdb_id
            poster_record = TMDbPoster.objects.filter(tmdb_id=tmdb_record.tmdb_id).first()
            if poster_record and poster_record.poster_path:
                return poster_record.poster_path

            # Fetch from the TMDb API if no poster record exists or it doesn't have a poster_path
            response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdb_record.tmdb_id}?api_key=bb74221ae90248cbc87b1360be4ee33e')
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                
                if poster_path:  # Check if poster_path is not None or empty
                    # Save the new poster path in TMDbPoster if a poster path is found
                    TMDbPoster.objects.create(tmdb_id=tmdb_record, poster_path=poster_path)
                    return poster_path

        except TMDb.DoesNotExist:
            return None
        return None
    
    def get_movie_slug(self, obj):
        # This new method fetches the slug from the MovieSlug model
        try:
            movie_slug = MovieSlug.objects.get(movie_id=obj)  # Assuming a direct relationship
            return movie_slug.movie_slug  # Return the slug field of the MovieSlug model
        except MovieSlug.DoesNotExist:
            return None  # Return None if no slug exists for the movie
        
    def get_tmdb_id(self, obj):
        try:
            tmdb_record = TMDb.objects.get(movie_id=obj.movie_id)
            return tmdb_record.tmdb_id
        except TMDb.DoesNotExist:
            return None