import numpy as np
import pandas as pd
from pathlib import Path
from ml_implement.general_utils.data_read import DataReader    

class StocasticGD:
    def __init__(self,learning_rate = None,epochs = None ):
        self.lr = learning_rate if learning_rate is not None else 0.01
        self.epochs = epochs if epochs is not None else 1000
        self.w = None
        self.b = 0
        
    def fit(self,X,y):
        m,d = X.shape
        if self.w is None:
            self.w = np.zeros(d)
            print(f"size of w just after initializing with np.zeros:-----------> {self.w.shape}")
        # self.w = np.zeros((d,1))  # initializing weights to zero vector
        for epoch in range(self.epochs):
            # suffle the Xin ecah epoch.
            indices  = np.random.permutation(m) # it will create the random number of sequence between 0(include) to m(not included).
            # indices0_4 = np.random.permutation(5) # o/p like this--> [4 0 3 1 2] ecah time it will be different but b/w 0-5(not included).

            X_suffle = X[indices] # X_suffle :--> rows of X is suffled randomly.
            y_suffle = y[indices] # similarly y row also suffled randomly each epoch.

            for i in range(m):

                # Pick ONE sample (Vectorized as 1xFeatures: X_suufle[i] -> is vector of size 1xFeatures )
                x_i = X_suffle[i]
                y_i = y_suffle[i]
                # 4. Predict (Dot product: Scalar result)
                y_hat = np.dot(x_i,self.w) + self.b

                # error is a scalar
                error = y_hat - y_i #  Compute Gradients (Matrix Form)

                dw = x_i*error # dw = x_i * error (Result is a vector of shape n_features)

                db = error
                if (i == 1) and (epoch == 1 or epoch == self.epochs) or (i == m-1)and(epoch == 1 or epoch == self.epochs):
                    print(f"y_hat size::---------->{y_hat.shape}")
                    print(f"error size:----------> {error.shape}")
                    print(f"dw size:---------->{dw.shape}")
                    print(f"db size:---------->{db.shape} and its type: {type(db)}")

            # Update Parameters 
                self.w -= self.lr*dw
                self.b -= self.lr*db      
        
    def predict(self,X):
        # X --> X_test # Full Matrix Multiplication for prediction: Y = Xw + b
        y_predict_dot_method=  np.dot(X, self.w) + self.b
        y = X@self.w + self.b
        return y,y_predict_dot_method
    