# ## Here we train the model using the train data and test the model using the test data and save the model in the model directory

# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd
# import numpy as np
# from dataclasses import dataclass
# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor


# from src.utils import save_object, evaluate_models # It is used to save the preprocessor object and to load the preprocessor object


# @dataclass

# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts','model.pkl')  # It is used to give the input to the model trainer
    
# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()
        
#     def initiate_model_trainer(self,train_array,test_array):
        
#         try:
#             logging.info("Split training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
            
#             models = {
                
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "K-Neighbors Regressor": KNeighborsRegressor(),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False)
#             }
            
            
#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)  # It is used to evaluate the model and get the best model score
            
#             ## To get the best model score from the dictionary
#             best_model_score = max(sorted(model_report.values()))
            
            
#             ## To get the best model name from the dictionary
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
            
#             best_model = models[best_model_name]
            
#             if best_model_score < 0.6:
#                 raise CustomException("No best model found")
            
#             logging.info("Best found model on both training and testing dataset")
            
            
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )
            
#             predicted = best_model.predict(X_test)
            
#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            
#         except Exception as e:
#             raise CustomException(e, sys)
#             logging.info("Error occurred while splitting training and test data")
            



# ## Homework : Do HyperParameter Tuning for the models and get the best hyperparameters for each model and then get the best model using the best hyperparameters and then save the best model in the model directory. Do it in such a way that it should not take much time to train the model. You can use RandomizedSearchCV for hyperparameter tuning. You can also use GridSearchCV for hyperparameter tuning but it will take more time to train the model. You can also use Optuna for hyperparameter tuning but it will take more time to train the model. You can also use Hyperopt for hyperparameter tuning but it will take more time to train the model. You can also use Bayesian Optimization for hyperparameter tuning but it will take more time to train the model. You can also use Genetic Algorithm for hyperparameter tuning but it will take more time to train the model. You can also use Particle Swarm Optimization for hyperparameter tuning but it will take more time to train the model. You can also use Simulated Annealing for hyperparameter tuning but it will take more time to train the model. You can also use Evolutionary Algorithm for hyperparameter tuning but it will take more time to train the model. You can also use Differential Evolution for hyperparameter tuning but it will take more time to train the model. You can also use Nelder-Mead for hyperparameter tuning but it will take more time to train the model. You can also use Powell's method for hyperparameter tuning but it will take more time to train the model. You can also use Conjugate Gradient for hyperparameter tuning but it will take more time to train the model. You can also use BFGS for hyperparameter tuning but it will take more time to train the model. You can also use L-BFGS-B for hyperparameter tuning but it will take more time to train the model. You can also use TNC for hyperparameter tuning but it will take more time to train the model. You can also use COBYLA for hyperparameter tuning but it will take more time to train the model. You can also use SLSQP for hyperparameter tuning but it will take more time to train the model. You can also use trust-constr for hyperparameter tuning but it will take more time to train the model. You can also use dogleg for hyperparameter tuning but it will take more time to train the model. You can also use trust-ncg for hyperparameter tuning but it will take more time to

import os
import sys
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
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
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
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
