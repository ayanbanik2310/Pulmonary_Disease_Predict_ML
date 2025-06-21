from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('final_model.pkl')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Example: Get input values from the form
        input1 = int(request.form['input1'])
        input2 = float(request.form['input2'])
        input3 = int(request.form['input3'])
        input4 = int(request.form['input4'])
        input5 = int(request.form['input5'])
        input6 = int(request.form['input6'])
        input7 = int(request.form['input7'])
        input8 = int(request.form['input8'])
        features = [[input1, input2, input3, input4, input5, input6, input7, input8]]
        
        prediction = model.predict(features)

        # Interpret the prediction
        if prediction[0] == 1:
            result = 'Pulmonary Disease Detected'
        else:
            result = 'No Pulmonary Disease'

        # Render the result in the template
        return render_template('index.html', prediction_text=f' {result}')


if __name__ == '__main__':
    app.run(debug=True)
