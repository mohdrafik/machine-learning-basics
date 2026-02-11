import numpy as np
import pandas as pd
from pathlib import Path
from ml_implement.general_utils.data_read import DataReader    

class Mine_GradientDescent_LinaerRgression:
    def __init__(self,learning_rate = None,epochs =None,initial_b =None, initial_w =None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b = initial_b
        self.w = initial_w
        self.J_w_b_history = {"w":[], "b":[], "J_cost":[]}
        # self.J_cost_hsitory = []

    def fit(self,X,y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        X = X.reshape(-1, X.shape[1])  # Ensure X is 2D
        # print(f"self.w: {self.w} and type of self.w: {type(self.w)} and isscalar: {np.isscalar(self.w)}")
        m,d = X.shape
        if self.w is None or self.w  == 0 or np.isscalar(self.w):   
            self.w = np.zeros((d,1))
        # self.w = np.zeros((d,1))  # initializing weights to zero vector of shape (d,1)
        # print(f"d : {d} - {self.w.shape} initial w: {self.w.flatten()} and shape of w: {self.w.shape} and initial b: {self.b}")

        for epoch in range(self.epochs):
            
            unitV = np.ones((m,1))    
            y = y.reshape(-1,1)
            # print(f" shape of the unitV: {unitV.shape} and Transpose of unitV: {(unitV.T).shape}")
            
            y_hat = X@self.w + self.b*unitV  # prediction using the hypothesis, written in matrix form so, 
            # it can work for single features as well as muultiple features. and cover the m no of training example as well as for single traing example.

            error = y_hat - y   # this term is common in both the gradient of b,w.[Xw + b*1 - y]-> [X(mxd)W(dx1) + b*1(1Xm unit vector) - y(mx1))]
            # self.J_cost_hsitory["J_cost"].append(np.mean(error**2))  # cost function history for each epoch.
            # grad_b = np.mean(error) or 1/m*unitV.T @ error or 1/m*np.dot(unitV.T,error)
            # grad_b = 1/m*unitV.T@ (X@self.w + self.b*unitV - y)
            # grad_b = np.mean(error)
            # grad_b = 1/m*np.dot(unitV.T,error)
            grad_b =  1/m*unitV.T @ error
            self.b = self.b - self.learning_rate*grad_b

            # grad_w = 1/m*X.T @ (X@self.w +self.b*unitV - y)
            grad_w = 1/m*X.T@ error
            self.w = self.w - self.learning_rate*grad_w
            # print(f"Epoch: {epoch+1}/{self.epochs},coefficient (w): {self.w.flatten()} intercept_b: {self.b} ")
            self.J_w_b_history["w"].append(self.w.flatten())
            self.J_w_b_history["b"].append(self.b)
            self.J_w_b_history["J_cost"].append(np.mean(error**2))
            
        return self.b,self.w,self.J_w_b_history 
    
    def predict(self,X_test):
        X_test = np.asarray(X_test, dtype=float)
        unitV1 = np.ones((X_test.shape[0],1))
        y_pred = unitV1*self.b + X_test@self.w
        
        return y_pred

if __name__== "__main__":
    filename  = "placement.csv"
    filepath = "/home/mrafiku/AI_learning/machine-learning-basics/data/Linear_Regression/placementdata"
    reader = DataReader(filepath=filepath, filename=filename,df_want=False,split=True)

    X_train, X_test, y_train, y_test = reader.run()
    print(f" < for check < -------------------> shape of X_train: {X_train.shape} and shape of y_train: {y_train.shape}")
    print(f" < for check < -------------------> shape of X_test: {X_test.shape} and shape of y_test: {y_test.shape}")

    mylr = Mine_GradientDescent_LinaerRgression(learning_rate = 0.01, epochs = 1000, initial_b = 0, initial_w = 0)
    mylr.fit(X_train,y_train)   
    mylr.predict(X_test)
    
    