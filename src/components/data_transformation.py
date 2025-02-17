from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
import sys


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation, including handling ordinal variables using Ordinal Encoding.
        '''
        try:
            
            numerical_columns = ["No_of_Bedrooms", "No_of_Bathrooms", "Flat_Area", "Lot_Area", "No_of_Floors",
                                 "Overall_Grade", "Area_of_the_House_from_Basement", "Basement_Area", "Age_of_House", 
                                 "Zipcode", "Latitude", "Longitude", "Living_Area_after_Renovation", "Lot_Area_after_Renovation", "Years_Since_Renovation"]
            nominal_categorical_columns = [
                "Waterfront_View", "Ever_Renovated"
            ]
            ordinal_categorical_columns = ["Condition_of_the_House" ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_nominal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ordinal_encoder", OrdinalEncoder()),
                ]
            )

            logging.info(f"Nominal Categorical columns: {nominal_categorical_columns}")
            logging.info(f"Ordinal Categorical columns: {ordinal_categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_nominal_pipeline", cat_nominal_pipeline, nominal_categorical_columns),
                    ("cat_ordinal_pipeline", cat_ordinal_pipeline, ordinal_categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def rename_columns(self, df):
        column_mapping = {
            "No of Bedrooms": "No_of_Bedrooms",
            "No of Bathrooms": "No_of_Bathrooms",
            "Flat Area (in Sqft)": "Flat_Area",
            "Lot Area (in Sqft)": "Lot_Area",
            "No of Floors": "No_of_Floors",
            "Overall Grade": "Overall_Grade",
            "Area of the House from Basement (in Sqft)": "Area_of_the_House_from_Basement",
            "Basement Area (in Sqft)": "Basement_Area",
            "Age of House (in Years)": "Age_of_House",
            "Zipcode": "Zipcode",
            "Latitude": "Latitude",
            "Longitude": "Longitude",
            "Living Area after Renovation (in Sqft)": "Living_Area_after_Renovation",
            "Lot Area after Renovation (in Sqft)": "Lot_Area_after_Renovation",
            "Waterfront View": "Waterfront_View",
            "Condition of the House": "Condition_of_the_House",
            "Renovated Year": "Renovated_Year",
            "Date House was Sold": "Date_House_was_Sold",
            "Sale Price": "Sale_Price",
            "No of Times Visited": "No_of_Times_Visited",
        }
        df = df.rename(columns=column_mapping)
        return df
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Rename columns in both train and test datasets")

            train_df = self.rename_columns(train_df)
            test_df = self.rename_columns(test_df)

            logging.info(f"Treating the missing values in target variable.")

            # Drop rows with missing target values
            train_df = train_df.dropna(subset=["Sale_Price"])
            test_df = test_df.dropna(subset=["Sale_Price"])

            logging.info(
                f"Manipulating datetime variable in training dataframe and testing dataframe."
            )

            # New variable creation
            train_df['Ever_Renovated'] = np.where(train_df['Renovated_Year'] == 0, 'No', 'Yes')
            
            # Manipulating datetime variable
            train_df['Purchase_Year'] = pd.DatetimeIndex(train_df['Date_House_was_Sold']).year
            train_df['Years_Since_Renovation'] = np.where(
                train_df['Ever_Renovated'] == 'Yes',
                abs(train_df['Purchase_Year'] - train_df['Renovated_Year']),
                0
            )

            test_df['Ever_Renovated'] = np.where(test_df['Renovated_Year'] == 0, 'No', 'Yes')
            
            test_df['Purchase_Year'] = pd.DatetimeIndex(test_df['Date_House_was_Sold']).year
            test_df['Years_Since_Renovation'] = np.where(
                test_df['Ever_Renovated'] == 'Yes',
                abs(test_df['Purchase_Year'] - test_df['Renovated_Year']),
                0
            )

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Sale_Price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Removing unnecessary columns from training dataframe and testing dataframe."
            )

            input_feature_train_df = input_feature_train_df.drop(columns=['No_of_Times_Visited', 'ID'], axis=1)
            input_feature_test_df = input_feature_test_df.drop(columns=['No_of_Times_Visited', 'ID'],axis=1)


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]



            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
