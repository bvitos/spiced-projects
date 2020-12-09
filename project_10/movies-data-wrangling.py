#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:58:36 2020

@author: boti
"""

import pandas as pd
import numpy as np

movies = pd.read_csv('./movies.csv',sep=',')
idlist = movies['movieId'].tolist

# Fix incomplete or wrongly formatted titles:

movies.loc[movies['title'].str[-1] == " ",'title']=movies['title'].str[0:-1]
movies.loc[6059,'title']="Babylon 5 (1998)"
movies.loc[9031,'title']="Ready Player One (2018)"
movies.loc[9091,'title']="Hyena Road (2015)"
movies.loc[9138,'title']="The Adventures of Sherlock Holmes and Doctor Watson (1980)"
movies.loc[9179,'title']="Nocturnal Animals (2016)"
movies.loc[9259,'title']="Paterson (2016)"
movies.loc[9367,'title']="Moonlight (2016)"
movies.loc[9448,'title']="The OA (2016)"
movies.loc[9514,'title']="Cosmos (2019)"
movies.loc[9515,'title']="Maria Bamford: Old Baby (2017)"
movies.loc[9525,'title']="Generation Iron 2 (2017)"
movies.loc[9611,'title']="Black Mirror (2018)"
movies['year'] = movies['title'].str[-5:-1]
movies['year'] = pd.to_numeric(movies['year'])
movies['alt_title'] = movies['title'].str[0:-7]
movies['title']=movies['alt_title']
for i in range(len(movies)):    # Separate alternative title from title (where applicable)
    if movies.loc[i,'title'][-1] == ")":
        movies.loc[i,'title'] = movies.loc[i,'title'][0:movies.loc[i,'title'].find("(")-1]
        movies.loc[i,'alt_title'] = movies.loc[i,'alt_title'][movies.loc[i,'alt_title'].find("(")+1:movies.loc[i,'alt_title'].find(")")]
        if movies.loc[i,'alt_title'][0:6] == "a.k.a.":
            movies.loc[i,'alt_title']=movies.loc[i,'alt_title'][6:len(movies.loc[i,'alt_title'])+1]

movies.to_csv('movies-wrangled.csv')