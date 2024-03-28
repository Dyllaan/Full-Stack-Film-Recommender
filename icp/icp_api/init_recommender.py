from icp_api.rs_models.collaborative_based_recommender import CFRecommender
from icp_api.rs_models.content_based_recommender import CBFRecommender
from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from keras.layers import Embedding, Flatten, Concatenate, Dense, StringLookup, Input
from tensorflow_recommenders.tasks import Ranking
from icp_api.models import Movie, Rating
from icp_api.rs_models.hybrid_recommender import CFRecommender

# Initialize the recommender system
#cf_recommender = CFRecommender()
#cf_recommender.startup()

# Initialise cbf

cbf_recommender = CBFRecommender()
cbf_recommender.startup()

cf_recommender = CFRecommender()
cf_recommender.startup()

def get_model():
    return cf_recommender