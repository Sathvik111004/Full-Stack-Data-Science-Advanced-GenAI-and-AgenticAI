from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Resolve path to the pickle file relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'linear_regression_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Simple HTML page with form and results
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Salary Prediction App</title>
</head>
<body>
    <h2>Salary Prediction App (Flask Version)</h2>
    <p>This app predicts the salary based on years of experience using a simple linear regression model.</p>
    
    <form method="POST" action="/">
        <label for="years_experience">Enter years of experience (0.0 to 50.0):</label>
        <input type="number" step="0.1" min="0" max="50" name="years_experience" id="years_experience" required value="{{ years_experience }}">
        <button type="submit">Predict Salary</button>
    </form>

    {% if error %}
        <p style="color: red;">{{ error }}</p>
    {% endif %}

    {% if prediction is not none %}
        <h3>Prediction Result:</h3>
        <p>The predicted salary for <strong>{{ years_experience }}</strong> years of experience is: <strong>${{ "{:,.2f}".format(prediction) }}</strong></p>
    {% endif %}

    <br>
    <small>Model built by prakash senapati. Deployed using Flask.</small>
</body>
</html>'''

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    years_experience = ""
    error = None
    
    if request.method == 'POST':
        if model is None:
            error = "Error: The model file could not be loaded on the server."
            return render_template_string(HTML_TEMPLATE, error=error, years_experience=years_experience, prediction=prediction)
            
        try:
            raw_input = request.form.get('years_experience', '')
            years_experience = float(raw_input)
            
            # Input validation constraints
            if years_experience < 0.0 or years_experience > 50.0:
                error = "Validation Error: Years of experience must be between 0.0 and 50.0."
            else:
                experience_input = np.array([[years_experience]])
                pred = model.predict(experience_input)
                prediction = float(pred[0])
        except ValueError:
            error = "Validation Error: Please enter a valid decimal number."
        except Exception as e:
            error = f"Error during prediction: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, error=error, years_experience=years_experience, prediction=prediction)

# Set secure headers in after_request
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

if __name__ == '__main__':
    # Listen only on localhost for development security
    app.run(host='127.0.0.1', port=5000, debug=True)