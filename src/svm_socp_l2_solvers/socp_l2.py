import numpy as np
from cvxopt import matrix, solvers
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from .utils import prediction_from_w_b,prediction_probas_from_w_b

class SOCPL2(BaseEstimator, ClassifierMixin):

    r"""
    Smoothed sparse L2-SOCP classifier.

    This estimator solves the following optimization problem:

    .. math::
        \min_{w,b,\xi}\ ||w||^2 \;+\; C\sum_{i=1}^2 \xi_i
        \quad \mathrm{s.t.}\quad
        \begin{aligned}
			&({\bf w}, b, \xi) \in \mathbb{R}^{n+2} \\
			&\text{s.t. } \ w^\top \mu_1 + b \geq 1 - \xi + \kappa(\alpha_1) \|S_1^\top w\|, \\
			&\quad -(w^\top \mu_2 + b) \geq 1 - \xi + \kappa(\alpha_2) \|S_2^\top w\|, \\
			&\quad \xi \geq 0
		\end{aligned}

    The vector :math:`\mu_1` (resp. :math:\mu_2) is the mean value vector of features associated with positive (resp. negative) class.
    The matrix :math:S_j\in\mathbb{R}^{n\times m_j}, with :math:j\in\{1,2\}, satisfy \sigma_j=S_jS_j^\top, where \sigma_1 (resp. \sigma_2) is the covariance matrix of features asociated with positive (resp. negative) class.   

    The constraint set of the above optimization problem is obtained from the following constraint set thanks to the the multivariate Chebyshev inequality:

    .. math::

    \inf_{\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} \!\!\! \text{Pr}\{(-1)^{j+1}({\bf w}^{\top }\widetilde{\bf x}_{j}+b)\ge 0\} \geq \alpha_j, \ j=1,2, 

    The notation :math:\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} means that the distributions :math:\widetilde{\bf x}_j have
    associated means and covariance matrices :math:({\bm\mu}_j, \Sigma_j) for :math:j = 1, 2.

    It is a robust version with respecto to noise of SVML2.

    The smoothing parameter :math:`\varepsilon>0` makes the objective locally
    Lipschitz and avoids singular behavior at :math:`w_j=0`.

    Parameters
    ----------
    p : float, default=0.5
        Exponent controlling sparsity. Must satisfy 0 < p < 1.

    C : float, default=1e4
        Slack penalty parameter. Must be > 0.

    alpha_1 : float, default=0.5
              Exponent controlling probability of good classification of positive class. Must satisfy 0 < alpha_1 < 1.

    alpha_2 : float, default=0.5   
              Exponent controlling probability of good classification of negative class. Must satisfy 0 < alpha_2 < 1.
              
    epsilon : float, default=1e-5
        Smoothing/approximation parameter :math:`\varepsilon>0` used in
        :math:`(|w_j|+\varepsilon)^p`.

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
                 

    Notes
    -----
    The problem is nonconvex given that p < 1; the solver may converge to a local
    minimum depending on the parameters.

    Example 
    -----

    from svm_socp_l2_solvers import SOCPL2
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    socp = SOCPL2(p=0.1,alpha_1=0.2,alpha_2=0.2)
    socp.fit(X,y)

    print("Coefs : ",socp.coef_)
    print("Selected features : ",socp.selected_feature_names_)

    """
    

    def __init__(self,C=1e4,alpha_1=0.5,alpha_2=0.5,tau = None,eps=1e-5,\
                 tol = 1e-3,max_iter = 100):
        

        self._C = None
        self.C = C
        self._alpha_1 = None
        self.alpha_1 = alpha_1
        self._alpha_2 = None 
        self.alpha_2 = alpha_2
        self._tau = None
        self.tau = tau   
        self._eps = None
        self.eps = eps
        self._tol = None
        self.tol = tol
        self._max_iter = None
        self.max_iter = max_iter
        
#        self.kappa1 = np.sqrt(alpha_1 / (1-alpha_1))
#        self.kappa2 = np.sqrt(alpha_2 / (1-alpha_2))


    @property 
    def C(self):
       return self._C

    @property 
    def alpha_1(self):
       return self._alpha_1

    @property 
    def alpha_2(self):
       return self._alpha_2    

    @property 
    def tau(self):
       return self._tau        
    
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

    @alpha_1.setter
    def alpha_1(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("alpha_1 must be a float number.")
        elif (value<=0) or (value>=1):
            raise ValueError("alpha_1 must be a real number between 0 and 1")
        else:
            self._alpha_1 = value
            self.kappa1 = np.sqrt(value / (1-value))

    @alpha_2.setter
    def alpha_2(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("alpha_2 must be a float number.")
        elif (value<=0) or (value>=1):
            raise ValueError("alpha_2 must be a real number between 0 and 1")
        else:
            self._alpha_2 = value  
            self.kappa2 = np.sqrt(value / (1-value))

    @tau.setter
    def tau(self,value):
        if value:
           if not isinstance(value, float) and not isinstance(value,int):
             raise TypeError("tau must be a float number or be equal to None.")
           elif (value<=0):
             raise ValueError("tau must be >0 and <=1")
           else:
             self._tau = value  
        else:
            self._tau = None               

    @eps.setter
    def eps(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("eps must be a float number or an integer number.")
        elif (value<=0):
            raise ValueError("eps must be a positive number")
        else:
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
            raise TypeError("max_iter must be an integer number.")
        elif (value<=0):
            raise ValueError("max_iter must be positive")
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
        Binary labels. Recommended: {-1, +1} or {0,+1}

        Returns
        -------
        self : object
        Fitted estimator.
        """        

        y = y.copy()
        X = X.copy()

        if hasattr(X,"columns"):
            self.feature_names_in_  = X.columns.tolist()

        
        X = check_array(X,force_all_finite=True)

        _ =  check_array(y,force_all_finite=True,ensure_2d=False)
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

        n = X.shape[1]

        self.n_features_in_ = n
        
        A_pos = X[(y==1).reshape((-1,))]
        A_neg = X[(y<=0).reshape((-1,))]
        
        m_pos = A_pos.shape[0]
        m_neg = A_neg.shape[0]

        
        mu1 = (1 / m_pos) * A_pos.T@np.ones((m_pos,1))
        mu2 = (1 / m_neg) * A_neg.T@np.ones((m_neg,1))
        
        S1 = (1 / np.sqrt(m_pos)) * (A_pos.T - mu1 @ np.ones((1,m_pos)))
        S2 = (1 / np.sqrt(m_neg)) * (A_neg.T - mu2 @ np.ones((1,m_neg)))

        G1 = np.concatenate([np.zeros((2,n+1)),-np.eye(2)],axis = 1)
        G2 = np.concatenate([-mu1.T,np.array([[-1,-1,0]])],axis = 1)
        G3 = np.concatenate([-self.kappa1*S1.T,np.zeros((m_pos,3))],axis = 1)
        G4 = np.concatenate([mu2.T,np.array([[1,0,-1]])],axis = 1)
        G5 = np.concatenate([-self.kappa2*S2.T,np.zeros((m_neg,3))],axis = 1)

        G_full = np.concatenate([G1,G2,G3,G4,G5],axis = 0)
        print(G_full.shape)
        G = matrix(G_full)

        h_full = np.concatenate([np.zeros((2,1)),-np.ones((1,1)),np.zeros((m_pos,1)),-np.ones((1,1)),np.zeros((m_neg,1))],axis = 0)
        h = matrix(h_full)

        dims = {'l':2,'q':[m_pos+1,m_neg+1],'s':[]}
        P11 = np.eye(n)
        P12 = np.zeros((n,3))
        P21 = P12.T
        P22 = np.zeros((3,3))
        P1 = np.concatenate([P11,P12],axis = 1)
        P2 = np.concatenate([P21,P22],axis = 1)
        P_full = np.concatenate([P1,P2],axis = 0)
        P = matrix(P_full)
        q_full = self.C * np.concatenate([np.zeros((n+1,1)),np.ones((2,1))],axis = 0)
        q = matrix(q_full)
        sol = solvers.coneqp(P=P, q=q, G=G, h=h, dims=dims)
        x = np.array(sol['x'])
        self.coef_ = x[:-3]
        self.intercept_ = x[-3]
        
        self.x = x
    
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
        
       if hasattr(self,"coef_") == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg)

       predictions =  prediction_from_w_b(self.coef_,self.intercept_,\
                                          X,threshold,self.negative_value)    
    
       return predictions
    
    def predict_proba(self,X):
       
       """
       Predict probability for class labels for samples in X.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)

       Returns
       -------
       y_pred_prob : ndarray of shape (n_samples,2)
        The first column is the probability for each observation to belong to 
        negative or zero class, the second column is the probability for each observation to belong to positive class.
       """    

       X = X.copy() 

       if hasattr(self,"coef_") == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg) 
       
       probas = prediction_probas_from_w_b(w=self.coef_,b=self.intercept_,X=X)
    
       return probas   
