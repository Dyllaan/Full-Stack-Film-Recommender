from rest_framework import serializers
from icp_api.models import Rating
from icp_api.api_serializers.movieserializer import MovieSerializer
import requests

class RatingSerializer(serializers.ModelSerializer):
    movie = MovieSerializer(read_only=True)
    class Meta:
        model = Rating
        fields = '__all__'

    def to_representation(self, instance):
        # Override the to_representation method to customize the output.
        representation = super().to_representation(instance)
        # If you want to flatten the structure and directly include 'movie_slug' at the top level:
        movie_slug = representation['movie'].get('movie_slug')
        representation['movie_slug'] = movie_slug  # Add 'movie_slug' to the representation
        del representation['movie']  # Optionally remove the nested 'movie' dictionary
        # If you keep the 'movie' dictionary, you don't need to delete it, and it will include all movie details.
        return representation