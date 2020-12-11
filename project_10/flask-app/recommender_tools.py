#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Recomender tools - Create NMF matrices - Find movies based on cosine similarity

"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from fuzzywuzzy import process
from scipy.spatial.distance import cdist
from threading import Thread
import time
import logging
from sklearn.metrics import mean_squared_error

class recommender:

    def __init__(self, movies, Q_movies):
        self.movies = movies
        self.Q_movies = Q_movies
        self.blacklist = []

    def fix_article_position(self, inputtitle):
        '''moves any articles (A/An/The) to the end of the title to correspond with database entries'''
        if inputtitle[0:2] == 'A ':
            inputtitle = inputtitle[2:] + ', A'
        if inputtitle[0:3] == 'An ':
            inputtitle = inputtitle[3:] + ', An'
        if inputtitle[0:4] == 'The ':
            inputtitle = inputtitle[4:] + ', The' + ''


    def find_movie_title(self, inputtitle, year):
        ''' validates the movie title/year using the fuzzywuzzy string matching algorithm'''
        if inputtitle == '':
            movietitle = ''
            movieyear = year
        else:
            inputtilte = self.fix_article_position(inputtitle)
            if year == "":
                choices = np.unique(self.movies['title'].tolist() + self.movies['alt_title'].tolist())
            else:
                choices = np.unique(self.movies[self.movies['year'] == int(year)]['title'].tolist() + self.movies[self.movies['year'] == int(year)]['alt_title'].tolist())
            movietitle = process.extractOne(inputtitle, choices)[0]
            if movietitle in self.movies['title'].tolist():
                movieyear = self.movies[self.movies['title']==movietitle]['year'].values[0]
            else:
                movieyear = self.movies[self.movies['alt_title']==movietitle]['year'].values[0]        
        return [movietitle, movieyear]


    def find_movie_index(self, inputtitle, year):
        ''' searches the movie index for given movie title/year using the fuzzywuzzy string matching algorithm'''
        inputtilte = self.fix_article_position(inputtitle)
        if year == "":
            choices = np.unique(self.movies['title'].tolist() + self.movies['alt_title'].tolist())
        else:
            choices = np.unique(self.movies[self.movies['year'] == int(year)]['title'].tolist() + self.movies[self.movies['year'] == int(year)]['alt_title'].tolist())
        movietitle = process.extractOne(inputtitle, choices)[0]
        if movietitle in self.movies['title'].tolist():
            return self.movies[self.movies['title']==movietitle]['movieId'].values[0]
        else:
            return self.movies[self.movies['alt_title']==movietitle]['movieId'].values[0]


    def movie_recommender(self, usrtitles, usryears):      
        '''uses the Q matrix to recommend movies that are closest to the individual movies provided, 
        plus another one that is closest to the mean of their features, based on cosine similarity,
        plus another one from which the sum of cosine distances is the lowest'''
        Q_work = self.Q_movies.copy(deep=True)
        recommendations = []
        indexes = []
        for i in range(len(usrtitles)):              # find and store the indexes of the usr movies
            indexes.append(str(self.find_movie_index(usrtitles[i], usryears[i])))
        for i in range(len(indexes)):
            usr_features = Q_work[indexes[i]]                       # store the features of usr move
            Q_work.drop(columns=indexes[i], inplace=True)          # remove usr movie from matrix
            distances = cdist(Q_work.T, [usr_features], 'cosine').flatten().tolist()   # calculate cosine distances
            min_index = np.argmin(distances)
            while Q_work.columns[min_index] in self.blacklist:       # disregard blacklisted (i.e. already recommended) movies
                distances[min_index] = 2
                min_index = np.argmin(distances)
            calcindex = Q_work.columns[min_index]
            recommendations.append(calcindex)    # retreive the column ID with the shortest distance in the cosine distances matrix
            Q_work.drop(columns=calcindex, inplace=True)          # remove usr movie from matrix
        distances = []
        for i in range(len(indexes)):
            cosine_distances = cdist(Q_work.T, [self.Q_movies[indexes[i]]], 'cosine').flatten().tolist()   # calculate cosine distances from usr vectors to all other vectors
            distances.append(cosine_distances)
        sum_distances = []
        for i in range(len(Q_work.columns)):            # another recommendation based on cumulative cosine distances
            sum_distances.append(sum(row[i] for row in distances))  # calculate sum of distances
        min_index = np.argmin(sum_distances)
        while Q_work.columns[min_index] in self.blacklist:       # disregard blacklisted (i.e. already recommended) movies
            sum_distances[min_index] = 2
            min_index = np.argmin(sum_distances)
        calcindex = Q_work.columns[min_index]   # retreive the column ID with the lowest sum of distances
        recommendations.append(calcindex)    
        Q_work.drop(columns=calcindex, inplace=True)          # remove usr movie from matrix
        combined_features = self.Q_movies[indexes].mean(axis=1)  # store the combined (mean) features of the usr movie
        distances = cdist(Q_work.T, [combined_features], 'cosine').flatten().tolist()   # "combined feature" recommendation: calculate cosine distances
        min_index = np.argmin(distances)
        while Q_work.columns[min_index] in self.blacklist:       # disregard blacklisted (i.e. already recommended) movies
            distances[min_index] = 2
            min_index = np.argmin(distances)
        recommendations.append(self.Q_movies.columns[min_index])    # retreive the column ID with the shortest distance in the cosine distances matrix    
        
    #        print("FOR LOOP:")                     # tested it with a for looop: same results, much slower
    #        shortest_distance = 1
    #        for column in Q_work.columns:
    #            distance = cdist([usr_features[i]], [Q_work[column]], 'cosine')
    #            if distance < shortest_distance:
    #                recommendation = column
    #                shortest_distance = distance
        logging.critical(recommendations)    
        recommended_movies=[]
        self.blacklist += recommendations
        for i in range(len(recommendations)):
            recommended_movies.append(self.movies[self.movies['movieId'] == int(recommendations[i])]['title'].values[0] + " (" + str(self.movies[self.movies['movieId'] == int(recommendations[i])]['year'].values[0]) + ")")
        return recommended_movies


def generate_matrix(ratingsread):
    ''' NMF matrix decomposition algorithm '''
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


if __name__ == "__main__":                 # When module is run directly it decomposes the ratings matrix and runs a test recommendation
    ratings = pd.read_csv('./ratings.csv',sep=',')
    P,Q,R = generate_matrix(ratings)
    Q.to_csv('./q.csv')
#    recommender.read_matrix()
    recommend = recommender(pd.read_csv('./movies-wrangled.csv',sep=','),pd.read_csv('./q.csv',sep=','))
    recmovies2=recommend.movie_recommender(['Heat', 'Nixon', 'Interiors'],[1995, 1995, 1978])