import os
import numpy as np
import pandas as pd
from pathlib import Path

class DataReader:
    def __init__(self,filepath, filename = None, filetype = None):
        self.filepath = Path(filepath)
        self.filename = filename
        self.filetype = filetype if filetype else filename.split('.')[-1]

    def read_data(self):
        if self.filetype == 'csv':
            fullfilepath = os.path.join(self.filepath,self.filename)
            return pd.read_csv(fullfilepath)
        elif self.filetype in ['xls', 'xlsx']:
            fullfilepath = os.path.join(self.filepath,self.filename)
            return pd.read_excel(fullfilepath)
        elif self.filetype == 'json':
            fullfilepath = os.path.join(self.filepath,self.filename)
            return pd.read_json(fullfilepath)
        elif self.filetype in ['txt', 'log']:
            fullfilepath = os.path.join(self.filepath,self.filename)
            with open(fullfilepath, 'r') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file type: {self.filetype}")
    
    def read_data_as_numpy(self):
        if self.filetype is not ['txt', 'log']:
            df = self.read_data()
            df.columns = df.columns.str.strip()
            print(f"df_head: {df.head()},df.columns: {df.columns}")
            return df
        else:
            df = self.read_data()
            return np.array(df.splitlines())
        
    def make_numpy_array(self):
        data = self.read_data_as_numpy()
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Data format not supported for conversion to numpy array.")

