from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and preprocessors
model = load_model('model/ipl_score_predictor_model.h5')
venue_encoder = joblib.load('model/venue_encoder.pkl')
batting_team_encoder = joblib.load('model/batting_team_encoder.pkl')
bowling_team_encoder = joblib.load('model/bowling_team_encoder.pkl')
striker_encoder = joblib.load('model/striker_encoder.pkl')
bowler_encoder = joblib.load('model/bowler_encoder.pkl')
scaler = joblib.load('model/scaler.pkl')

# Load dataset to extract unique values for dropdowns
df = pd.read_csv('ipl_data.csv')
venues = sorted(df["venue"].dropna().unique())
batting_teams = sorted(df["bat_team"].dropna().unique())
bowling_teams = sorted(df["bowl_team"].dropna().unique())
strikers = sorted(df["batsman"].dropna().unique())
bowlers = sorted(df["bowler"].dropna().unique())

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        venue = request.form['venue']
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        striker = request.form['striker']
        bowler = request.form['bowler']

        # Encode and scale
        try:
            input_data = np.array([
                venue_encoder.transform([venue])[0],
                batting_team_encoder.transform([bat_team])[0],
                bowling_team_encoder.transform([bowl_team])[0],
                striker_encoder.transform([striker])[0],
                bowler_encoder.transform([bowler])[0]
            ]).reshape(1, -1)

            input_scaled = scaler.transform(input_data)

            # Predict
            predicted_score = model.predict(input_scaled)[0][0]
            predicted_score = int(predicted_score)

            return render_template('index.html',
                                   prediction=predicted_score,
                                   venues=venues,
                                   batting_teams=batting_teams,
                                   bowling_teams=bowling_teams,
                                   strikers=strikers,
                                   bowlers=bowlers)
        except Exception as e:
            return render_template('index.html',
                                   error=str(e),
                                   venues=venues,
                                   batting_teams=batting_teams,
                                   bowling_teams=bowling_teams,
                                   strikers=strikers,
                                   bowlers=bowlers)

    return render_template('index.html',
                           venues=venues,
                           batting_teams=batting_teams,
                           bowling_teams=bowling_teams,
                           strikers=strikers,
                           bowlers=bowlers)
                           
if __name__ == '__main__':
    app.run(debug=True)
