import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from cvxpy.error import SolverError
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from .utils import prediction_from_w_b,prediction_probas_from_w_b

class SVML2(BaseEstimator, ClassifierMixin):

    r"""
    Smoothed sparse Lp-SVM classifier.

    This estimator solves the following optimization problem:

    .. math::

        \min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p \;+\; C\sum_{i=1}^m \xi_i
        \quad \mathrm{s.t.}\quad
        y_i (w^\top x_i + b) \geq 1 - \xi_i,\ \xi_i \geq 0,\ i=1,\dots,m.

    The smoothing parameter :math:`\varepsilon>0` makes the objective locally
    Lipschitz and avoids singular behavior at :math:`w_j=0`.

    Parameters
    ----------

    C : float, default=1e4
        Slack penalty parameter. Must be > 0.

    eps : float, default=1e-5
        Smoothing/approximation parameter :math:`\varepsilon>0` used in

    tol : float, default=1e-4
        Tolerance for stopping criteria.         
        
    max_iter : int, default=100
        Maximum iterations for converging


    Methods
    -------
    fit(X, y)
        Fit the model on labeled data.

    predict(X)
        Predict class labels for samples in X.

    predict_proba(X)
        Estimate probability of the positive class.


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.

    coef_ : ndarray of shape (n_features,)
        Estimated weight vector.

    intercept_ : float
        Estimated intercept.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of detected features after calling fit()

    feature_names_in_ : ndarray of shape (n_classes,)
            Names of features seen during :term:`fit`. Defined only when `X` has feature names that are all strings.

                 


    Example 
    -----

    from svm_socp_l2_solvers import SVML2
    import pandas as pd

    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-l2-solvers/main/datos_Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    svm = SVML2(C = 1e7,eps = 1e-4)
    svm.fit(X,y)

    print("Coefs : ",svm.coef_)
    print("Selected features : ",svm.selected_feature_names_)


    """
    

    def __init__(self,C=1e4,eps=1e-5,tol=1e-4,max_iter=100):

        
        self._C = None
        self.C = C 
        self._eps = None
        self.eps = eps 
        self._tol = None
        self.tol = tol
        self._max_iter = None
        self.max_iter = max_iter
        self.eps = eps
        self._tol = None
        self.tol = tol
        self._max_iter = None
        self.max_iter = max_iter      

    

    @property 
    def C(self):
       return self._C
  
    
    @property
    def eps(self):
        return self._eps
    

    @property
    def tol(self):
        return self._tol



    @property
    def max_iter(self):
        return self._max_iter        



    @C.setter
    def C(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("C must be a float number.")
        elif (value<=0):
            raise ValueError("C must be a positive number")
        else:
            self._C = value

    @eps.setter
    def eps(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("eps must be a float number.")
        elif (value<=0):
            raise ValueError("eps must be a positive number")
        else:

            self._eps = value  

            self._eps = value


    @tol.setter
    def tol(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("tol must be a float number or an integer number.")
        elif (value<=0):
            raise ValueError("tol must be a positive number")
        else:
            self._tol = value 

    @max_iter.setter
    def max_iter(self,value):
        if not isinstance(value,int):
            raise TypeError("max_iter must be a float number.")
        elif (value<=0):
            raise ValueError("max_iter must be a positive number")
        else:
            self._max_iter = value                            
               

    def fit(self,X,y):


        """
        Fit the Lp-SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Training data.

        y : array-like of shape (n_samples,)
        Binary labels. Must be {-1, +1} or {0,+1}

        Returns
        -------
        self : object
        Fitted estimator.
        """

        y = y.copy()
        X = X.copy()

        if hasattr(X,"columns"):
            self.feature_names_in_ = X.columns.tolist()
        
        X = check_array(X,ensure_all_finite=True)

        _ =  check_array(y,ensure_all_finite=True,ensure_2d=False)
        if isinstance(y,np.ndarray) == False:
            y = np.array(y)
            
        y = y.astype(float)
        
        self.negative_value = y.min()  
        
        
        if y.ndim == 2:
            if y.shape[1] > 1:
                raise ValueError("y's number of columns must be equal to one")

        
        y = y.reshape((-1,1))
        if X.shape[0] != y.shape[0]:
            raise ValueError("The dimensions of X and y are not consistent")
            
        if (len(np.unique(y)) != 2):
            raise ValueError("The target must be a binary variable.")

        if (set(np.unique(y)) != {0,1}) & (set(np.unique(y)) != {-1,1}):
            raise ValueError("The target must contain only -1 and 1 or 0 and 1.")
            
        self.classes_ = np.unique(y)
        y[y<=0] = -1

        m = X.shape[0]
        n = X.shape[1]

        self.n_features_in_ = n
        
        w_old = np.random.randn(n)

        phi_k = np.ones(n)
        err = 2 * self.tol
        iter_ = 0
        # ========= Variables =========
        w  = cp.Variable(n)
        b  = cp.Variable()
           
        xi =  cp.Variable(m,nonneg=True)
        constraints = [] 
        for row, target,xi_i in zip(X,y,xi):
            constr = target @ (w @ row.reshape((-1,1)) + b) >=  1 - xi_i
            constraints.append(constr) 


        self.n_non_zeros_coefs_per_iteration_ = []    
 
        while (err > self.tol and iter_ < self.max_iter):    
            
           weighted_abs = cp.multiply(phi_k, w) 
           obj = cp.Minimize(cp.norm2(weighted_abs)**2 + self.C * cp.sum(xi)) 
           # ========= Resolver =========
           prob = cp.Problem(obj, constraints)
           prob.solve()   
           err = npl.norm(w.value - w_old,np.inf) 
           w_old = w.value
           b_old = b.value
           xi_old = xi.value
           phi_k = np.sqrt(self.p/2)*(np.abs(w_old)**2+self.eps) ** ((self.p-2)/4)        

           self.n_non_zeros_coefs_per_iteration_.append(int((np.abs(w_old) > \
                                                            self.tol_select_features).sum()))                    

           iter_ += 1
            
        self.coef_ = w_old
        self.intercept_ = b_old
        self.xi = xi_old 
        self.fitted_ = True

        self.n_iter_ = iter_ 

        self.n_iter_ = iter_
        self.n_non_zeros_coefs_per_iteration_ = np.array(self.n_non_zeros_coefs_per_iteration_)


        mask_selected_features = np.abs(w_old) > self.tol_select_features
        self.n_selected_features_ = int(mask_selected_features.sum())


        if hasattr(self,"feature_names_in_"):
            self.selected_feature_names_ = np.array(self.feature_names_in_)[mask_selected_features]
        
    def predict(self,X,threshold = 0.5): 

       """
    Predict class labels for samples in X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted labels in the same encoding as `classes_`.
       """          

       X = X.copy() 
        
       if self.fitted_ == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg)

       predictions =  prediction_from_w_b(self.coef_,self.intercept_,\
                                          X,threshold,self.negative_value)    
    
       return predictions
    
    def predict_proba(self,X):
       
       """
    Predict probabilities for samples in X to belong to positive class.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    y_proba : ndarray of shape (n_samples,2)
        Predicted probabilites in the same encoding as `classes_`.
       """  
       X = X.copy() 

       if self.fitted_ == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg) 
       
       probas = prediction_probas_from_w_b(w=self.coef_,b=self.intercept_,X=X)
    
       return probas   
