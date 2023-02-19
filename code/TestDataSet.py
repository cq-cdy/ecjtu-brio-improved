from torch.utils.data import Dataset
import utils
import os
class ValidDataSet(Dataset):
    def __init__(self,dataset_path,l=5):
        self.path = dataset_path
        self.len1 = l
        

    def __getitem__(self, item):
        file_path = os.path.join(self.path,f'{item}.json')
        data = utils.json_read(file_path)
        return data['x'],data['label']
    
    def __len__(self):
        return self.len1

class TestDataSet(Dataset):
    def __init__(self,dataset_path):
        self.path = dataset_path
        self.len = os.listdir(self.path)
        self.len =[i for i in self.len if i.endswith('.json')]

    def __getitem__(self, item):
        file_path = os.path.join(self.path,f'{item}.json')
        data = utils.json_read(file_path)
        return data['x'],data['label']
    def __len__(self):
        return len(self.len)