# File name: main_stage1_models.py
# Description: Fit stage 1 models on historic streaming history
#              using playlist and likability score outcomes,
#              save models and vectors of selected features        
# Author: Chris Rowe

import numpy as np
import pandas as pd
import os 
from sklearn.ensemble import GradientBoostingClassifier                            
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import pickle

def main():

    # import data
    os.chdir(os.path.dirname(os.getcwd()))
    X = pd.read_csv('data/processed/X.csv').drop(columns = ['artistName', 'trackName'])
    y_score = pd.read_csv('data/processed/y_score.csv').values.ravel()
    y_playlist = pd.read_csv('data/processed/y_playlist.csv').values.ravel()

    # initialize hyperparameter grids for random search
    xgb_param_grid={'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [100, 500, 1000],
                    'subsample': [1.0, 0.75, 0.5],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [3, 5, 7],
                    'max_features': [None, 'sqrt']} 

    # playlist score model (classification)

    # feature selection:

    # initialize model and random search
    xgb_cls_fs = RandomizedSearchCV(GradientBoostingClassifier(), 
                                    xgb_param_grid, n_iter = 50, 
                                    n_jobs = -1, cv = 10, 
                                    random_state = 6053)

    # execute search
    print("Fitting playlist models for feature selection...")
    xgb_cls_fs.fit(X, y_playlist)

    # subset features based on feature importance
    playlist_feat_selector = SelectFromModel(xgb_cls_fs.best_estimator_, prefit=True)
    X_fs_playlist = playlist_feat_selector.transform(X)
    playlist_features = playlist_feat_selector.get_support()

    # main model
    # initialize model and random search
    xgb_cls = RandomizedSearchCV(GradientBoostingClassifier(), 
                                 xgb_param_grid, n_iter = 50, 
                                 n_jobs = -1, cv = 10, 
                                 random_state = 6053)

    # execute search
    print("Fitting final playlist models...")
    xgb_cls.fit(X_fs_playlist, y_playlist)

    # print best score
    print("Final playlist model:")
    print(xgb_cls.best_score_) 
    print(xgb_cls.best_params_)

    # playlist score model (classification)

    # feature selection:

    # initialize model and random search
    xgb_reg_fs = RandomizedSearchCV(GradientBoostingRegressor(), 
                                    xgb_param_grid, n_iter = 50, 
                                    n_jobs = -1, cv = 10, 
                                    random_state = 6053)

    # execute search
    print("Fitting likeability score models for feature selection...")
    xgb_reg_fs.fit(X, y_score)

    # subset features based on feature importance
    score_feat_selector = SelectFromModel(xgb_reg_fs.best_estimator_, prefit=True)
    X_fs_score = score_feat_selector.transform(X)
    score_features = score_feat_selector.get_support()

    # main model
    # initialize model and random search
    xgb_reg = RandomizedSearchCV(GradientBoostingRegressor(), 
                                 xgb_param_grid, n_iter = 50, 
                                 n_jobs = -1, cv = 10, 
                                 random_state = 6053)

    # execute search
    print("Fitting final likeability score models...")
    xgb_reg.fit(X_fs_score, y_playlist)

    # print best score
    print("Final likeability score model:")
    print(xgb_reg.best_score_) 
    print(xgb_reg.best_params_)


    # save models and training features

    # playlist model
    filename_playlist = 'saved_models/xgb_playlist_model_wg.sav'
    pickle.dump(xgb_cls.best_estimator_, open(filename_playlist, 'wb'))

    # score model
    filename_score = 'saved_models/xgb_score_model_wg.sav'
    pickle.dump(xgb_reg.best_estimator_, open(filename_score, 'wb'))

    # training_features
    training_features_playlist = pd.DataFrame(list(X.columns[playlist_features]))
    training_features_playlist.to_csv('training_features/stage1_playlist_training_features.csv',
                                      header=False, index=False)
    training_features_score = pd.DataFrame(list(X.columns[score_features]))
    training_features_score.to_csv('training_features/stage1_score_training_features.csv',
                                   header=False, index=False)

if __name__ == '__main__':
    main()
