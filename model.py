import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

# Prediction function
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(request.form['duration']),
                    float(request.form['director_fb_likes']),
                    float(request.form['actor_1_fb_likes']),
                    float(request.form['gross']),
                    float(request.form['num_voted_users']),
                    float(request.form['facenumber_in_poster']),
                    float(request.form['budget']),
                    float(request.form['title_year']),
                    float(request.form['aspect_ratio']),
                    float(request.form['movie_fb_likes']),
                    float(request.form['Other_actor_fb_likes']),
                    float(request.form['critic_review_ratio']),
                    float(request.form['imdb_binned_score']),
                    float(request.form['country_USA']),
                    float(request.form['country_other']),
                    float(request.form['content_rating_NC-17']),
                    float(request.form['content_rating_PG']),
                    float(request.form['content_rating_PG-13']),
                    float(request.form['content_rating_R']),
                    float(request.form['content_rating_TV-G']),
                    float(request.form['content_rating_TV-PG']),
                    float(request.form['content_rating_UR'])]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted IMDB binned score: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
