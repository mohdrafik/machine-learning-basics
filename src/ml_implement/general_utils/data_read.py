import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataReader:
    """
        Cleans column names.
        Returns both:
        df (pandas DataFrame),
        X (features),                                           
        y (target),
        Includes a method to split data into train/test sets only if requested. Supports flexible target column selection.
        Keeps modular structure for reuse.
    Example usage:
        reader = DataReader(filepath="data", filename="example.csv")
        df, X_train, X_test, y_train, y_test = reader.train_test(target_column="Label", split=True)
    
    """

    def __init__(self, filepath, filename=None, filetype=None):
        self.filepath = Path(filepath)
        self.filename = filename
        self.filetype = filetype if filetype else (filename.split('.')[-1] if filename else None)
        # self.fullfilepath = self.filepath / self.filename if self.filename else None

    def read_data(self):
        fullfilepath = self.filepath / self.filename

        if not fullfilepath.exists():
            raise ValueError(f"File does not exist at path: {fullfilepath}")


        if self.filetype == 'csv':
            df =  pd.read_csv(fullfilepath)
            df.columns = df.columns.str.strip()  # clean column names
            print(f" DataFrame Head:\n{df.head()}")
            print(f" Columns: {list(df.columns)}")
            return df
        elif self.filetype in ['xls', 'xlsx']:
            df = pd.read_excel(fullfilepath)
            df.columns = df.columns.str.strip()  # clean column names
            print(f" DataFrame Head:\n{df.head()}")
            print(f" Columns: {list(df.columns)}")
            return df
        elif self.filetype == 'json':
            df = pd.read_json(fullfilepath)
            df.columns = df.columns.str.strip()  # clean column names
            print(f" DataFrame Head:\n{df.head()}")
            print(f" Columns: {list(df.columns)}")
            return df
        elif self.filetype in ['txt', 'log']:
            with open(fullfilepath, 'r') as file:
                df = file.read()
                df = df.splitlines()
                print(f" Data:\n{df}")
            return df   

        else:
            raise ValueError(f"Unsupported file type: {self.filetype}")

    def make_numpy_array(self):
        data = self.read_data()
        if self.filetype in ['txt', 'log']:
            data = np.array(data) 

        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Data format not supported for conversion to numpy array.")

    # def get_features_targets(self, target_column=None):
    #     df = self.read_data()

    #     if not isinstance(df, pd.DataFrame):
    #         raise ValueError("Feature-target split is only valid for tabular data.")

    #     if target_column and target_column not in df.columns:
    #         raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    #     X = df.drop(columns=[target_column]) if target_column else df.iloc[:, :-1]
    #     y = df[target_column] if target_column else df.iloc[:, -1]
        
    #     print(f"✅ Features shape: {X.shape} | Target shape: {y.shape}")
    #     return df, X.values, y.values

    @staticmethod   
    def train_test(df,X,y,test_size=0.2, random_state=42):
        # df, X, y = self.get_features_targets(target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("📊 Data split complete:")
        print(f"   → X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   → X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        return X_train, X_test, y_train, y_test

if __name__=="__main__":
    filepath = Path(__file__).resolve().parent.parent.parent / "data"/ "Linear_Regression"/"chicago_houseprice"
    print(f"📁 Reading data from: {filepath}")
    filename = 'house_prices_dataset.csv'
    reader = DataReader(filepath = filepath, filename=filename)
    df = reader.read_data()
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = reader.train_test(df,X,y,test_size=0.2, random_state=42)
    print(X_train[1:4,:], X_test[1:4,:], y_train[1:4], y_test[1:4])