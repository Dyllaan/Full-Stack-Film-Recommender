import csv
import random
from faker import Faker
import psycopg2
from psycopg2 import sql

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="icp",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()

# Load ratings CSV file and extract user_ids
user_ids = set()
with open('ratings.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        user_ids.add(int(row['userId']))

# Initialize Faker to generate fake user data
fake = Faker()

# Define the number of fake users to generate
num_fake_users = len(user_ids)

# Generate fake user accounts
fake_users = []
for _ in range(num_fake_users):
    username = fake.user_name()
    first_name = fake.first_name()
    last_name = fake.last_name()
    email = fake.email()
    password = fake.password()
    fake_users.append((username, first_name, last_name, email, password))

# Insert fake users into the database
for user_id, (username, first_name, last_name, email, password) in zip(user_ids, fake_users):
    insert_query = sql.SQL("INSERT INTO icp_api_apiuser (id, username, first_name, last_name, email, password) VALUES (%s, %s, %s, %s, %s, %s)")
    cur.execute(insert_query, (user_id, username, first_name, last_name, email, password))

# Commit the transaction
conn.commit()

# Close cursor and connection
cur.close()
conn.close()

print("Fake users inserted successfully.")
