from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')  # Create an index.html file with your desired UI

@app.route('/predictions', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Create a home.html file for form submission
    else:
        data = CustomData(
            carat=float(request.form.get('carat')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity'),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=f"Predicted Price: ${results[0]:,.2f}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Run the application
