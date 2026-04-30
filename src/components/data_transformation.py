######  Feature engineering , data cleaning and data transformation ######


import pandas as pd
import numpy  as np
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

from src.utils import save_object, load_object  # It is used to save the preprocessor object and to load the preprocessor object

@dataclass                    
class DataTransformationConfig:  # it is used to store the configuration of data transformation
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')  # It is used to give the input to the data transformation
    
    
    
class DataTransformation:
    """
    This function is responsible for data transformation."""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # It is used to initialize the data transformation configuration
        
        
    def get_data_transformer_object(self):  ## It is use to create the pikle files  used to convert categorical data into numerical
        
        try:
            
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(   ### It is used to handle missing values 
                 
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),  # It is used to impute the missing values with the median value
                    ("scaler",StandardScaler(with_mean=False))  # It is used to scale the numerical values
                    
                ]
            )
                
            cat_pipeline=Pipeline(  ### It is used to handle categorical values
                                      
                         steps=[
                             ("imputer",SimpleImputer(strategy="most_frequent")),  # It is used to impute the missing values with the most frequent value
                             ("one_hot_encoder",OneHotEncoder()),  # It is used to convert categorical values into numerical values
                             ("scaler",StandardScaler(with_mean=False))  # It is used to scale the categorical values
                             
                         ]  
                ) 
            
            logging.info("Numerical columns standard scaling completed")
            
            logging.info("Categorical columns encoding completed")    
            
            
            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),  # It is used to apply the numerical pipeline to the numerical columns
                    ("cat_pipeline",cat_pipeline,categorical_columns)  # It is used to apply the categorical pipeline to the categorical columns
                ]
                
            )     
            return preprocessor 
            
        except Exception as e:
            
            raise CustomException(e,sys)
        
        
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)    #  It is used to fit the preprocessing object on the training data and transform the training data
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)     # It is used to transform the test data using the same preprocessing object that was fitted on the training data
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)  # It is used to concatenate the input features and target feature of the training data
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)  # It is used to concatenate the input features and target feature of the test data
            ]
            
            ## .c_ is used to concatenate the input features and target feature of the training data and test data
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,  # It is used to give the input to the data transformation
                obj=preprocessing_obj  # It is used to save the preprocessing object)
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,  # It is used to return the path of the preprocessor object file
            )
        except Exception as e:
            raise CustomException(e,sys)
        
