from django.contrib.auth.models import Group
from rest_framework import serializers
from icp_api.models import ApiUser
from icp_api.models import Movie
from icp_api.models import Rating
from django.contrib.auth.hashers import make_password
from icp_api.api_serializers.movieserializer import MovieSerializer

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiUser
        fields = ['username', 'first_name', 'last_name', 'date_of_birth', 'email', 'password']
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def create(self, validated_data):
        return ApiUser.objects.create_user(**validated_data)


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']
    
class RatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Rating
        fields = '__all__'
