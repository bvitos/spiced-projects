#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:50:59 2020

Recomender tools - Create NMF matrices - Find movies based on cosine similarity

"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from fuzzywuzzy import process
from scipy.spatial.distance import cdist
import time
import logging
    
def generate_matrix(ratingsread):
    ratingsread.drop(columns=['timestamp'],inplace=True)
    ratings = pd.pivot_table(ratingsread, values = 'rating', index='userId',columns='movieId')
    averagerating = ratings.mean().mean()
#    ratingsimp = ratings.fillna(averagerating)
    ratingsimputed = ratings.fillna(0)
    nmf = NMF(n_components=40, max_iter=2000)
    nmf.fit(ratingsimputed)
    Q = nmf.components_
    Q = pd.DataFrame(Q, columns=ratings.columns)
    P = nmf.transform(ratingsimputed)
    error = round(nmf.reconstruction_err_, 2)
    R = pd.DataFrame(np.dot(P, Q), columns=ratings.columns, index=ratings.index)
    return P,Q,R


def fix_article_position(inputtitle):
    '''moves articles a/an/the to the end of the title to correspond with database entries'''
    if inputtitle[0:2] == 'A ':
        inputtitle = inputtitle[2:] + ', A'
    if inputtitle[0:3] == 'An ':
        inputtitle = inputtitle[3:] + ', An'
    if inputtitle[0:4] == 'The ':
        inputtitle = inputtitle[4:] + ', The' + ''


def find_movie_title(inputtitle, year):
    ''' finds the movie with the most similar title to usr input title from a given year'''
    if inputtitle == '':
        movietitle = ''
        movieyear = year
    else:
        inputtilte = fix_article_position(inputtitle)
        if year == "":
            choices = np.unique(movies['title'].tolist() + movies['alt_title'].tolist())
        else:
            choices = np.unique(movies[movies['year'] == int(year)]['title'].tolist() + movies[movies['year'] == int(year)]['alt_title'].tolist())
        movietitle = process.extractOne(inputtitle, choices)[0]
        if movietitle in movies['title'].tolist():
            movieyear = movies[movies['title']==movietitle]['year'].values[0]
        else:
            movieyear = movies[movies['alt_title']==movietitle]['year'].values[0]        
    return [movietitle, movieyear]


def find_movie_index(inputtitle, year):
    ''' provides the index of the movie with the most similar title from a given year'''
    inputtilte = fix_article_position(inputtitle)
    if year == "":
        choices = np.unique(movies['title'].tolist() + movies['alt_title'].tolist())
    else:
        choices = np.unique(movies[movies['year'] == int(year)]['title'].tolist() + movies[movies['year'] == int(year)]['alt_title'].tolist())
    movietitle = process.extractOne(inputtitle, choices)[0]
    if movietitle in movies['title'].tolist():
        return movies[movies['title']==movietitle]['movieId'].values[0]
    else:
        return movies[movies['alt_title']==movietitle]['movieId'].values[0]


def movie_recommender(usrtitles, usryears):      
    '''recommends movies with features closest to the individual movies provided, 
    plus the one closest to the mean of their features, based on cosine similarity'''
    Q_work = Q_movies.copy(deep=True)
    recommendations = []
    indexes = []
    for i in range(len(usrtitles)):              # find and store the indexes of the usr movies
        indexes.append(str(find_movie_index(usrtitles[i], usryears[i])))
    combined_features = Q_work[indexes].mean(axis=1)  # store the combined (mean) features of the usr movie
    for i in range(len(indexes)):
        usr_features = Q_work[indexes[i]]         # store the features of usr move
        Q_work.drop(columns=indexes[i], inplace=True)          # remove usr movie from matrix
        distances = cdist(Q_work.T, [usr_features], 'cosine')   # calculate cosine distances
        calcindex = Q_work.columns[np.argmin(distances)]
        recommendations.append(calcindex)    # retreive the column ID with the shortest distance in the cosine distances matrix
        Q_work.drop(columns=calcindex, inplace=True)          # remove usr movie from matrix
    distances = cdist(Q_work.T, [combined_features], 'cosine')   # "combined feature" recommendation: calculate cosine distances
    recommendations.append(Q_work.columns[np.argmin(distances)])    # retreive the column ID with the shortest distance in the cosine distances matrix    
    
#        print("FOR LOOP:")                     # tested it with a for looop: same results, much slower
#        shortest_distance = 1
#        for column in Q_work.columns:
#            distance = cdist([usr_features[i]], [Q_work[column]], 'cosine')
#            if distance < shortest_distance:
#                recommendation = column
#                shortest_distance = distance

    recommended_movies=[]
    for i in range(len(recommendations)):
        recommended_movies.append(movies[movies['movieId'] == int(recommendations[i])]['title'].values[0] + " (" + str(movies[movies['movieId'] == int(recommendations[i])]['year'].values[0]) + ")")

    return recommended_movies


def read_matrix():                      # initialising data, this will be run in a separate thread
    global movies
    movies = pd.read_csv('./movies-wrangled.csv',sep=',')
    global Q_movies
    Q_movies = pd.read_csv('./q.csv',sep=',')

if __name__ == "__main__":                 # When module is run directly it generates the Q matri and runs a test recommendation
    P,Q,R = generate_matrix(pd.read_csv('./ratings.csv',sep=','))
    Q.to_csv('./q.csv')
    read_matrix()
    recmovies=movie_recommender(['Heat', 'Nixon', 'Interiors'],[1995, 1995, 1978])