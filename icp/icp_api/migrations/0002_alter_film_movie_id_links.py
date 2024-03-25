# Generated by Django 4.2.7 on 2024-02-23 14:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('icp_api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='film',
            name='movie_id',
            field=models.AutoField(db_column='movie_id', primary_key=True, serialize=False),
        ),
        migrations.CreateModel(
            name='Links',
            fields=[
                ('link_id', models.AutoField(primary_key=True, serialize=False)),
                ('imdb_id', models.CharField(db_column='imdb_id', max_length=15)),
                ('tmdb_id', models.IntegerField(db_column='tmdb_id')),
                ('movie_id', models.ForeignKey(db_column='movie_id', on_delete=django.db.models.deletion.CASCADE, to='icp_api.film')),
            ],
            options={
                'db_table': 'links',
            },
        ),
    ]