from pathlib import Path
import os
import numpy as np

def load_data():
    """  load numpy data """
    # root_dir = Path.cwd().parent.parent
    root_dir = Path(__file__).parent.parent.parent
    
    rel_path = "data/unsupervised_learning/"
    filename = 'ex7_X.npy'
    # print(f"{root_dir} and full filepath:{os.path.join(root_dir,rel_path)}")
    full_filepath = os.path.join(root_dir,rel_path)
    fullfile = os.path.join(full_filepath,filename)
    data = np.load(fullfile)
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    
    return data
if __name__ == "__main__":
    rel_path = "data/unsupervised_learning/"
    filename = 'ex7_X.npy'
    data = load_data()
    print(f"{data.shape}: ---> data is loaded successfully")