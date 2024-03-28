# Generated by Django 4.2.7 on 2024-03-28 02:02

from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    dependencies = [
        ('icp_api', '0020_auto_20240328_0201'),
    ]

    operations = [
        migrations.CreateModel(
            name='MovieSlug',
            fields=[
                ('slug_id', models.AutoField(db_column='slug_id', primary_key=True, serialize=False)),
                ('movie_id', models.ForeignKey(db_column='movie_id', on_delete=models.CASCADE, to='icp_api.Movie')),
                ('movie_slug', models.SlugField(max_length=255, unique=True)),
            ],
            options={
                'db_table': 'movie_slugs',
            },
        ),
    ]