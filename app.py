#app.py
from flask import Flask, request, jsonify, render_template
from food_model import FoodModel

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import re
import random

app = Flask(__name__)

df_recipes = pd.read_csv('data/recipes_reduce.csv')
df_reviews = pd.read_csv('data/reviews_reduce.csv')
df_reviews = df_reviews.astype({'AuthorId': 'string'})

Reviews = tf.data.Dataset.from_tensor_slices(dict(df_reviews[['AuthorId', 'Name', 'Rating']]))
Recipes = tf.data.Dataset.from_tensor_slices(dict(df_recipes[['Name']]))

Reviews = Reviews.map(lambda x: {
    "Name": x["Name"],
    "AuthorId": x["AuthorId"],
    "Rating": float(x["Rating"])
})

Recipes = Recipes.map(lambda x: x["Name"])

food_titles = Recipes.batch(1_000)
user_ids = Reviews.batch(1_000).map(lambda x: x["AuthorId"])

unique_food_titles = np.unique(np.concatenate(list(food_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

model = FoodModel(rating_weight=1.0,
                  retrieval_weight=1.0,
                  unique_user_ids=unique_user_ids,
                  unique_food_titles=unique_food_titles,
                  Recipes=Recipes)

dummy_input = {
    "AuthorId": [str(unique_user_ids[0])],
    "Name": [str(unique_food_titles[0])],
    "Rating": [0.0] 
}

for key in dummy_input:
    dummy_input[key] = tf.constant(dummy_input[key], dtype=tf.string if key != "Rating" else tf.float32)

model(dummy_input)

model.load_weights('model/food_recommendation_model.h5')


def recommend_food_for_random_user(model, recipe_df, top_n=5):
    random_user_id = df_reviews['AuthorId'].sample(1).values[0] # Pilih secara acak dari unique_user_ids
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((Recipes.batch(100), Recipes.batch(100).map(model.food_model)))
    )

    _, titles = index(tf.constant([str(random_user_id)]))

    recommended_titles = [title.decode("utf-8") for title in titles[0, :top_n].numpy()]
    return recommended_titles, random_user_id

# Function to get image URL for a given recipe name
def get_image_url(recipe_df, recipe_name, image_column='Images'):
    image_urls = recipe_df[recipe_df['Name'] == recipe_name][image_column].values
    if len(image_urls) > 0:
        match = re.search(r'https://.*?\.(jpg|JPG|jpeg)', image_urls[0])
        if match:
            return match.group()
    return None

# Route for the home page with recommendations
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    recommended_titles, random_user_id = recommend_food_for_random_user(model, df_recipes, top_n=5)

    for i, title in enumerate(recommended_titles):
        image_url = get_image_url(df_recipes, title)
        recommendations.append({
            'title': title,
            'image_url': image_url
        })

    return render_template('index.html', recommendations=recommendations, random_user_id=random_user_id)

if __name__ == '__main__':
    app.run(debug=True)

