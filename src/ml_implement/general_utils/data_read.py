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

    def __init__(self, filepath, filename=None, filetype=None,target_column = None,split=False,df_want = False,test_size=None, random_state=None):
        self.filepath = Path(filepath)
        self.filename = filename
        self.filetype = filetype if filetype is not None else filename.split('.')[-1] 
        self.split = split  # default to False
        self.df_want = df_want 
        self.target_column = target_column if target_column else None 
        self._get_target_column_interactively()
        self.test_size = test_size if test_size else 0.2
        self.random_state = random_state if random_state else 42

    
    def _get_target_column_interactively(self):
        if self.target_column is None:
            print(f"Warning Always check: if only last column is your target column then it is fine, otherwise specify the target column.")
            print(f"Filename: {self.filename}")
            print(f"Target column: {self.target_column}")
            user_input = input("Enter the target column name enter string value desired (or press Enter to use the last column): ").strip()
            if user_input:
                self.target_column = user_input
                print(f" Now Selected Target column ----> : {self.target_column}")

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
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Data format not supported for conversion to numpy array.")

    def get_features_targets(self, target_column=None):
        """
        Separates features (X) and target (y) from a DataFrame.
        
        Args:
            target_column (str, optional): The name of the target column. 
                                           If None, the last column is assumed to be the target.
        
        Returns:
            tuple: (df, X, y) where:
                - df: The original DataFrame
                - X: Feature matrix (numpy array)
                - y: Target vector (numpy array)
        """
        df = self.read_data()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Feature-target split is only valid for tabular data.")

        if target_column and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        X = df.drop(columns=[target_column]) if target_column else df.iloc[:, :-1]
        y = df[target_column] if target_column else df.iloc[:, -1] # by the deafult selecting the last column as target column.
        
        print(f" Features shape: {X.shape} | Target shape: {y.shape}")
        return df, X.values, y.values

    # @staticmethod   
    def train_test(self, df,X,y,test_size=0.2, random_state=42):  
        if self.split == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            print("📊 Data split complete:")
            print(f"   → X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"   → X_test:  {X_test.shape}, y_test:  {y_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def run(self):
        """
        Orchestrates the process: reads data, separates features/targets, 
        and optionally splits into train/test sets.
        
        Returns:
            tuple: Dependent on configuration:
                - If split=True: (X_train, X_test, y_train, y_test)
                - If split=False: (X, y)
                - If df_want=True: Prepend df to the above.
        """
        # 1. Get DataFrame, Features (X), and Target (y)
        print(f"output of class in this order: df, X_train, X_test, y_train, y_test ")
        print(f" Final Choosen Target column ----> : {self.target_column}")
        df, X, y = self.get_features_targets(target_column=self.target_column)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        # 2. Split data if requested (handled by train_test internally via self.split)
        data_split = self.train_test(df, X, y, test_size=self.test_size, random_state=self.random_state)
        
        # 3. Return results based on configuration
        if self.df_want:
            return (df,) + data_split
        else:
            return data_split



if __name__=="__main__":
    # Resolve project root: src/ml_implement/general_utils/data_read.py -> ... -> project_root
    project_root = Path(__file__).resolve().parent.parent.parent.parent 
    # filepath = project_root / "data" / "Linear_Regression" / "chicago_houseprice"
    filepath = project_root / "data" / "Linear_Regression" / "logistic_regressionData"
    
    print(f"📁 Reading data from: {filepath}")
    filename = 'cancer_data.csv'
    # filename = 'house_prices_dataset.csv'
    
    # Example: Wanting DataFrame and Train/Test Split, targeting 'price'
    # reader = DataReader(filepath=filepath, filename=filename, split=True, df_want=True, target_column="price")
    reader = DataReader(filepath=filepath, filename=filename, split=True, df_want=True, target_column=None)
    
    try:
        df, X_train, X_test, y_train, y_test = reader.run()
        print(f"✅ Success! Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Preview X_train:\n{X_train[:3]}")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
    