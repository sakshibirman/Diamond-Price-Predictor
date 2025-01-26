import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.train_data_path = os.path.join("data", "train.csv")  
        self.test_data_path = os.path.join("data", "test.csv")  

    def run_pipeline(self):
        try:
            # Load datasets
            logging.info("Loading training and testing data.")
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            # Data Transformation
            logging.info("Starting data transformation.")
            data_transformation = DataTransformation()
            train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=self.train_data_path, test_path=self.test_data_path
            )

            # Model Training
            logging.info("Starting model training.")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_array, test_array)

            logging.info(f"Training pipeline completed successfully. Model R2 Score: {r2_score:.2f}")
        
        except Exception as e:
            raise CustomException(e, sys)
