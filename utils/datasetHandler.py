import os
import pandas as pd

from dataDownload import DatasetDownloader

class DatasetSeperation:
    def __init__(self, csv_name):
        self.main_path = 'dataset'
        self.train_path = os.path.join(self.main_path,"train")
        self.valid_path = os.path.join(self.main_path,"valid")
        
        self.path()
        
        self.download_data = DatasetDownloader()
        self.raw_data = pd.read_csv(self.download_data.getPath()+f'/{csv_name}')
        self.create_train_valid_df()
    
    def path(self):
        if not os.path.exists(self.main_path):
            os.mkdir(self.main_path)
        if not os.path.exists(self.valid_path):
            os.mkdir(self.valid_path)
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
    
    def create_train_valid_df(self, valid_ratio=0.2, random_state=42):
        data_length = len(self.raw_data)
        
        valid_size = int(data_length * valid_ratio)
        
        fraud_df = self.raw_data[self.raw_data["Class"] == 1]
        
        non_fraud_df = self.raw_data[self.raw_data["Class"] == 0]
        
        remaining_valid_size = max(0, valid_size - len(fraud_df))
        
        valid_non_fraud = non_fraud_df.sample(n=remaining_valid_size, random_state=random_state)
        
        valid_dataset = pd.concat([fraud_df, valid_non_fraud]).reset_index(drop=True)
        
        train_dataset = self.raw_data.drop(valid_dataset.index).reset_index(drop=True)
        
        train_dataset.to_csv(os.path.join(self.train_path, "train.csv"), index=False)
        
        valid_dataset.to_csv(os.path.join(self.valid_path, "valid.csv"), index=False)
        
        return train_dataset, valid_dataset
