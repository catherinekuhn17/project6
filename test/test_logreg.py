import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler


#from utils import loadDataset
#from logreg import LogisticRegression

"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_updates():
    """
    this function tests if the gradiant and loss scores are being calculated correctly!
    """
    # load data and build model
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 
                                                           'Computed tomography of chest and abdomen', 
                                                           'Plain chest X-ray (procedure)',  
                                                           'Low Density Lipoprotein Cholesterol', 
                                                           'Creatinine', 'AGE_DIAGNOSIS'],
                                                             split_percent=0.8, split_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)


    log_model = logreg.LogisticRegression(num_feats=6, 
                                          max_iter=20000, 
                                          tol=.0001, 
                                          learning_rate=0.0001, 
                                          batch_size=200)

    log_model.train_model(X_train, y_train, X_val, y_val)

    # Check that your loss function is correct and that 
    # you have reasonable losses at the end of training
    
    # asserting that training loss has decreased during training
    assert log_model.loss_history_train[-1] < log_model.loss_history_train[0] 
    
    # asserting reasonable training loss value (have found a number of different parameters converge 
    # between .2 and .4, so checking loss at least gets to below .4
    assert log_model.loss_history_train[-1] < .4
    
    # Now checking that loss function is correct. To do this, we will look at only one itteration of the
    # algorithm. We will then compare using the the method of computing loss, with the known equation
    log_model = logreg.LogisticRegression(num_feats=6, 
                                          max_iter=2, 
                                          tol=.0001, 
                                          learning_rate=0.001, 
                                          batch_size=200)
    log_model.train_model(X_train, y_train, X_val, y_val)
    loss_score_method = log_model.loss_function(log_model.curr_x, log_model.curr_y)
    
    yp = log_model.pred
    N = log_model.curr_x.shape[0] # curr_x and curr_y are saved in the log_model class
    # known loss score equation
    loss_score_calc = (-1/N)*sum(
                        [(yi*np.log(ypi)+(1-yi)*np.log(1-ypi)) 
                        for yi, ypi in zip(log_model.curr_y, yp)] 
                      )


    # make sure this is equal to when we call it!
    assert loss_score_method == loss_score_calc
    
    # Now do the same for gradiant! 
    grad_method = log_model.calculate_gradient(log_model.curr_x, log_model.curr_y)
    diff = log_model.make_prediction(log_model.curr_x) - log_model.curr_y
    N = log_model.curr_x.shape[0]
    grad_calc = (np.dot(log_model.curr_x.T, diff))
    
    assert np.alltrue(grad_method == grad_calc)


def test_predict():
    """
    this function tests that W is being updated, and that the ultimate W found can accurately
    predict the train and validation sets
    """
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 
                                                           'Computed tomography of chest and abdomen', 
                                                           'Plain chest X-ray (procedure)',  
                                                           'Low Density Lipoprotein Cholesterol', 
                                                           'Creatinine', 'AGE_DIAGNOSIS'],
                                                           split_percent=0.8, 
                                                           split_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)


    log_model = logreg.LogisticRegression(num_feats=6, 
                                          max_iter=20000, 
                                          tol=.0001, 
                                          learning_rate=0.001, 
                                          batch_size=200)

    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # first, we want to assert that W is being updated in each round
    for i in range(1, len(log_model.track_W)):
        assert sum(abs(log_model.track_W[i]-log_model.track_W[i-1])) != 0
    
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
    # comparing predicted y agains actual y, for train and test data sets
    # for this I am using get_prediction, which uses make_prediction, and then assigns each yi to 0 or 1
    percent_wrong_train = sum(abs(y_train-log_model.get_prediction(X_train)))/len(y_train) 
    percent_wrong_test = sum(abs(y_val-log_model.get_prediction(X_val)))/len(y_val)

    assert percent_wrong_train < .2 # checking accuracy on training data
    assert percent_wrong_test < .2 # checking accuracy on testing data
