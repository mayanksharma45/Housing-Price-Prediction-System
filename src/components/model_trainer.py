import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [3, 5, 10, None]
                    },
                "Random Forest": {
                    'n_estimators': [10, 50, 100]  
                    },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute']
                    },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05],
                    'subsample': [0.6, 0.8],
                    'n_estimators': [10, 50, 100]  
                    },
                "Linear Regression": {},  
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [10, 50, 100]
                    },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],  
                    'learning_rate': [0.05, 0.1],
                    'iterations': [30, 50]  
                    },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [10, 50, 100]  
                    }
            }
            

            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param = params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            print(f"R Squared Value : {r2_square}")
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)