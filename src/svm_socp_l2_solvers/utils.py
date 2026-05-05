from sklearn.utils.validation import check_array
import numpy as np

def prediction_from_w_b(w,b,X,threshold,negative_value):

       X = check_array(X,force_all_finite=True)
            
       if X.shape[1] != len(w):
          raise ValueError(f"The model has been fitted with {len(w)} features.")
            
       p = 1 / (1 + np.exp(-X @ w.reshape((-1,1))-b))
       predictions = (p>threshold).reshape((-1,1)).astype(int)
       if negative_value == -1:
            predictions = 2 * (predictions - 0.5)

       return predictions

def prediction_probas_from_w_b(w,b,X):    

       X = check_array(X,force_all_finite=True)
            
       if X.shape[1] != len(w):
          raise ValueError(f"The model has been fitted with {len(w)} features.")
            
       p = 1 / (1 + np.exp(-X @ w.reshape((-1,1))-b))
       p = p.reshape((-1,1)) 
    
       return np.concatenate([1-p,p], axis = 1)  