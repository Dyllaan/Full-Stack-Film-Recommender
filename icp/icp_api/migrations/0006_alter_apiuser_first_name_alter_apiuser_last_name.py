# Generated by Django 4.2.7 on 2024-02-23 23:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('icp_api', '0005_alter_apiuser_date_of_birth_alter_apiuser_email'),
    ]

    operations = [
        migrations.AlterField(
            model_name='apiuser',
            name='first_name',
            field=models.CharField(max_length=30),
        ),
        migrations.AlterField(
            model_name='apiuser',
            name='last_name',
            field=models.CharField(max_length=30),
        ),
    ]
