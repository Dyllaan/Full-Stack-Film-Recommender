# views.py

import csv
from django.http import HttpResponse
from django.views import View
from icp_api.models import ApiUser
from faker import Faker

class GenerateFakeUsersView(View):
    def get(self, request):
        # Load ratings CSV file and extract unique user_ids
        user_ids = set()
        with open(r'C:\Users\Louis\Desktop\ICP\icp\icp_api\ml\ratings.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_ids.add(int(row['userId']))

        # Initialize Faker to generate fake user data
        fake = Faker()

        # Create users
        for user_id in user_ids:
            id = user_id  # Use the user_id as the primary key
            username = f'user_{user_id}'  # Generate a username based on the user_id
            first_name = fake.first_name()  # Generate a random first name
            last_name = fake.last_name()  # Generate a random last name
            email = self.generate_unique_email(fake)  # Generate a unique email address
            date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)  # Generate a random date of birth
            password = 'password'  # Set a default password for all users (change as needed)

            # Create a user object and save it to the database
            ApiUser.objects.create_user(id=id, username=username, first_name=first_name, last_name=last_name, email=email, password=password, date_of_birth=date_of_birth)

        return HttpResponse("Fake users inserted successfully.")

    def generate_unique_email(self, fake):
        """
        Generate a unique email address using Faker.
        If the email address already exists in the database, append a digit until a unique email is generated.
        """
        email = fake.email()
        while ApiUser.objects.filter(email=email).exists():
            # Extract the numeric part of the email address (before the '@' symbol)
            prefix, suffix = email.split('@')
            # Append a digit to the numeric part
            prefix += '1'
            # Reconstruct the email address
            email = f'{prefix}@{suffix}'
        return email
