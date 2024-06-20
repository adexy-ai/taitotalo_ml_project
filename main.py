from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample data for training the model
data = {
    'population': [1000, 2000, 3000, 4000, 5000],
    'diabetes_cases': [50, 100, 150, 200, 250]
}

df = pd.DataFrame(data)

X = df[['population']]
y = df['diabetes_cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    population = int(request.form['population'])
    prediction = model.predict(np.array([[population]]))[0]
    return jsonify({'predicted_diabetes_cases': prediction})

if __name__ == "__main__":
    app.run(debug=True)
