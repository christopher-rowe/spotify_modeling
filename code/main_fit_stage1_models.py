
import numpy as np
import pandas as pd
import os 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

def main():

    # import data
    os.chdir(os.path.dirname(os.getcwd()))
    X = pd.read_csv('data/processed/X.csv').drop(columns = ['artistName', 'trackName'])
    y_score = pd.read_csv('data/processed/y_score.csv').values.ravel()
    y_playlist = pd.read_csv('data/processed/y_playlist.csv').values.ravel()

    # initialize xgboost hyperparameters for random search
    xgb_param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5],
                    'n_estimators': [100, 500, 1000, 1500],
                    'subsample': [1.0, 0.75, 0.5],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [3, 5, 7, 9, 11],
                    'max_features': [None, 'sqrt']}


    # playlist score model (classification)
    # initialize model and random search
    xgb_cls = GradientBoostingClassifier()
    xgb_cls_rs = RandomizedSearchCV(xgb_cls, xgb_param_grid, n_iter = 100, 
                                    n_jobs = -1, cv = 10, 
                                    random_state = 6053)

    # execute search
    xgb_cls_rs.fit(X, y_playlist)

    # save optimal model
    xgb_playlist_model = xgb_cls_rs.best_estimator_

    # print best score
    xgb_cls_rs.best_score_, xgb_cls_rs.best_params_

    # likeability score model (regression)
    # initialize model and random search
    xgb_reg = GradientBoostingRegressor()
    xgb_reg_rs = RandomizedSearchCV(xgb_reg, xgb_param_grid, n_iter = 100, 
                                    n_jobs = -1, cv = 10, 
                                    random_state = 6053)

    # execute search
    xgb_reg_rs.fit(X, y_score)

    # save optimal model
    xgb_score_model = xgb_reg_rs.best_estimator_

    # print best score
    xgb_reg_rs.best_score_, xgb_reg_rs.best_params_

    # save models and training features

    # playlist model
    filename_playlist = 'saved_models/xgb_playlist_model_wg.sav'
    pickle.dump(xgb_playlist_model, open(filename_playlist, 'wb'))

    # score model
    filename_score = 'saved_models/xgb_score_model_wg.sav'
    pickle.dump(xgb_score_model, open(filename_score, 'wb'))

    # training_features
    training_features = pd.DataFrame(list(X.columns))
    training_features.to_csv('training_features/stage1_training_features.csv',
                            header=False, index=False)

if __name__ == '__main__':
    main()
