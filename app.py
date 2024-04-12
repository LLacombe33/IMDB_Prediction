import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
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
    output = (prediction[0])
    prediction_labels = {1: 'Flop', 2: 'Ok', 3: 'Good', 4: 'POGGERS'}   
    predicted_label = prediction_labels[output]
    return render_template('index.html', prediction_text='Predicted IMDB binned score: {}'.format(predicted_label))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
