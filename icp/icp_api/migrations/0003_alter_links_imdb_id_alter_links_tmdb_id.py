# Generated by Django 4.2.7 on 2024-02-23 14:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('icp_api', '0002_alter_film_movie_id_links'),
    ]

    operations = [
        migrations.AlterField(
            model_name='links',
            name='imdb_id',
            field=models.CharField(db_column='imdb_id', null=True),
        ),
        migrations.AlterField(
            model_name='links',
            name='tmdb_id',
            field=models.IntegerField(db_column='tmdb_id', null=True),
        ),
    ]
