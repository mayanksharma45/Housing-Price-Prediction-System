import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            # print("Predictions:", preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        No_of_Bedrooms: int,
        No_of_Bathrooms: float,
        Flat_Area: float,
        Lot_Area: float,
        No_of_Floors: float,
        Waterfront_View: str,
        Condition_of_the_House: str,
        Overall_Grade: int,
        Area_of_the_House_from_Basement: float,
        Basement_Area: int,
        Age_of_House: int,
        Zipcode: float, 
        Latitude: float, 
        Longitude: float,
        Living_Area_after_Renovation: float,
        Lot_Area_after_Renovation: int,
        Ever_Renovated: str,
        Years_Since_Renovation: float):

        self.No_of_Bedrooms = No_of_Bedrooms
        self.No_of_Bathrooms = No_of_Bathrooms
        self.Flat_Area = Flat_Area
        self.Lot_Area = Lot_Area
        self.No_of_Floors = No_of_Floors
        self.Waterfront_View = Waterfront_View
        self.Condition_of_the_House = Condition_of_the_House
        self.Overall_Grade = Overall_Grade
        self.Area_of_the_House_from_Basement = Area_of_the_House_from_Basement
        self.Basement_Area = Basement_Area
        self.Age_of_House = Age_of_House
        self.Zipcode = Zipcode
        self.Latitude = Latitude
        self.Longitude = Longitude
        self.Living_Area_after_Renovation = Living_Area_after_Renovation
        self.Lot_Area_after_Renovation = Lot_Area_after_Renovation
        self.Ever_Renovated = Ever_Renovated
        self.Years_Since_Renovation = Years_Since_Renovation

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "No_of_Bedrooms": [self.No_of_Bedrooms],
                "No_of_Bathrooms": [self.No_of_Bathrooms],
                "Flat_Area": [self.Flat_Area],
                "Lot_Area": [self.Lot_Area],
                "No_of_Floors": [self.No_of_Floors],
                "Waterfront_View": [self.Waterfront_View],
                "Condition_of_the_House": [self.Condition_of_the_House],
                "Overall_Grade": [self.Overall_Grade],
                "Area_of_the_House_from_Basement": [self.Area_of_the_House_from_Basement],
                "Basement_Area": [self.Basement_Area],
                "Age_of_House": [self.Age_of_House],
                "Zipcode": [self.Zipcode],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
                "Living_Area_after_Renovation": [self.Living_Area_after_Renovation],
                "Lot_Area_after_Renovation": [self.Lot_Area_after_Renovation],
                "Ever_Renovated": [self.Ever_Renovated],
                "Years_Since_Renovation": [self.Years_Since_Renovation],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
