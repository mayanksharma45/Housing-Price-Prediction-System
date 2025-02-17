from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            No_of_Bedrooms=int(request.form.get('No_of_Bedrooms')),
            No_of_Bathrooms=float(request.form.get('No_of_Bathrooms')),
            Flat_Area=float(request.form.get('Flat_Area')),
            Lot_Area=float(request.form.get('Lot_Area')),
            No_of_Floors=float(request.form.get('No_of_Floors')),
            Waterfront_View=str(request.form.get('Waterfront_View')),
            Condition_of_the_House=str(request.form.get('Condition_of_the_House')),
            Overall_Grade=int(request.form.get('Overall_Grade')),
            Area_of_the_House_from_Basement=float(request.form.get('Area_of_the_House_from_Basement')),
            Basement_Area=int(request.form.get('Basement_Area')),
            Age_of_House=int(request.form.get('Age_of_House')),
            Zipcode=float(request.form.get('Zipcode')),
            Latitude=float(request.form.get('Latitude')),
            Longitude=float(request.form.get('Longitude')),
            Living_Area_after_Renovation=float(request.form.get('Living_Area_after_Renovation')),
            Lot_Area_after_Renovation=int(request.form.get('Lot_Area_after_Renovation')),
            Ever_Renovated=str(request.form.get('Ever_Renovated')),
            Years_Since_Renovation=float(request.form.get('Years_Since_Renovation')),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        

