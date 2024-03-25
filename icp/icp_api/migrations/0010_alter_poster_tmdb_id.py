# Generated by Django 4.2.7 on 2024-03-04 22:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('icp_api', '0009_tmdb_alter_poster_tmdb_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='poster',
            name='tmdb_id',
            field=models.ForeignKey(db_column='tmdb_id', null=True, on_delete=django.db.models.deletion.CASCADE, to='icp_api.links'),
        ),
    ]
