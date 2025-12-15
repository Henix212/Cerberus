import kagglehub

class DatasetDownloader():
    
    def __init__(self):
        self.path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    def getPath(self):
        return self.path