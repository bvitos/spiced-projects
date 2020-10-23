#!/usr/bin/env python
# coding: utf-8

"""
Artist predictor based on lyrics
Scrapes track lyrics from lyrics.com and evaluates them for artist predictions
Matches contents of a text file with a pre-existing artist in the dataframe
@author: boti
"""

# Import pandas etc
import pandas as pd
import numpy as np
import sys
import requests
import time
from bs4 import BeautifulSoup
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Import spacy and load the language model
import spacy
nlp = spacy.load('en_core_web_md')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Define cleaning function
def clean_text(review, model):
    """preprocess a string (tokens, stopwords, lowercase, lemma & stemming) returns the cleaned result
        params: review - a string
                model - a spacy model
        returns: list of cleaned tokens
    """
    new_doc = ''
    doc = model(review)
    for word in doc:
        if not word.is_stop and word.is_alpha:                     # keeps alphabetic strings, excludes stopwords
            new_doc = f'{new_doc} {word.lemma_.lower()}'           # appends to result
            
    return new_doc



def gettrackurls(soup):
    """ 
        extracts track urls from the lyrics.com artist page
    """
    titles = soup.body.find_all('td', attrs={'class': 'tal qx'})
    links = []
    for title in titles:
        str = title.find('a', href=True)
        for tag in title.findAll('a', href=True):
            links.append('https://www.lyrics.com' + tag['href'])
    return links


#### Check for command line arguments

import argparse
parser = argparse.ArgumentParser(description='Artist/Lyrics predictor')
parser.add_argument('-l', '--lyrics', help="The text file containing the lyrics that should be identified", type = str)
parser.add_argument('-a', '--artistlinks', help='The text file containing the links of the artists pages on lyrics.com (each link on a separate row)', type = str)
#parser.add_argument('-a', '--artistlinks', nargs='+', help='The text file containing the links of the artists pages on lyrics.com (each link on a separate row)')



args = parser.parse_args()
lyrics_file=args.lyrics
#lyrics_file="lyrics-powerflo.txt"

# Instantiate the CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# !!!todo: save the model after corpus expansion with pickle. //load and use the model when identifying lyrics (just drop the new features when merging the dataframes; TfidfTransformer should come last then use model prediction on last row)

    
if args.artistlinks != None:
    
    df_vectorized_corpus = pd.read_csv('bagoflyrics.csv')     # Import existing vectorized data (X); artist names (y)
    X = df_vectorized_corpus.drop(columns='000')
    y = df_vectorized_corpus['000']    

    f = open(args.artistlinks, "r")         # Import artist links to be added
#    newartistlinks = f.readlines()
    url = f.read().splitlines()
    number_of_artists=len(url)

                                            # Start scraping:
    headers = {'headers': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.80 Safari/537.36"}
    artist=[]
    for i in range(number_of_artists):
        artistpage = requests.get(url[i], headers=headers)
        if artistpage.status_code == 200: 
            print('Artist page '+str(i + 1)+': OK!')
        elif artistpage.status_code == 404: 
            print('Artist page '+str(i + 1)+': not found!')

        cut1=url[i].rfind('/')
        cut0=url[i].rfind('/',0,cut1-1) + 1
        artistname=url[i][cut0:cut1]
        artistname = artistname.replace("-","+")
        if artistname in y.tolist():
            print(f"Artist {artistname} was already scraped, ignoring duplicate...")
        else:
            artist.append(artistpage.text)
            print(f"Artist name: {artistname}.")
        time.sleep(0.5)

    number_of_artists=len(artist)
    
    trackurls=[]
    for i in artist:
        soup = BeautifulSoup(i, "lxml")
        trackurls.append(gettrackurls(soup))

    dframe = pd.DataFrame(columns=['lyrics','artist'])

    print("Scraping lyrics...")
    lyrics=[]
    for i in range(number_of_artists):
        tracktitles = []
        artistlyrics = []
        cut1=trackurls[i][0].rfind('/')
        cut0=trackurls[i][0].rfind('/',0,cut1-1) + 1
        artistname=trackurls[i][0][cut0:cut1]
        artistname = artistname.replace("-","+")
        for j in range(len(trackurls[i])):
            lyricspage = requests.get(trackurls[i][j], headers=headers)
            if lyricspage.status_code == 200: 
                print('OK: ' + trackurls[i][j])
            elif lyricspage.status_code == 404: 
                print('404 Error: ' + trackurls[i][j])
            soup=BeautifulSoup(lyricspage.text, "lxml")
            currenttracktitle=soup.find(id="lyric-title-text")
            if currenttracktitle in tracktitles:
                print(f"Track {currenttracktitle} already scraped, ignoring duplicate...")
            else:                
                text=soup.find(id="lyric-body-text")
                artistlyrics.append(text.text)        
                newrow = {'lyrics' : text.text, 'artist': artistname}
                dframe = dframe.append(newrow, ignore_index=True)
                time.sleep(0.5)
                tracktitles.append(currenttracktitle)
        lyrics.append(artistlyrics)    
    df = pd.read_csv('lyrics.csv')         # read in the  lyrics database (scraped prior to running the program)
    df = df.append(dframe)
    df.to_csv('lyrics2.csv', index=False)
    print('Lyrics dataframe (lyrics.csv):')
    print(df)
    
#    sys.exit()


    df['lyrics'] = df['lyrics'].apply(clean_text, model=nlp)     # apply a function with keywords to df
    X = df['lyrics']
    y = df['artist']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=24)
    # Fit the CountVectorizer
    vectorizer.fit(Xtrain)
    # Transform the CountVectorizer
    vectorized_Xtrain = vectorizer.transform(Xtrain)
    # #### Sparse Matrix
    # Most of our matrix consists of zeroes. A Sparse Matrix only stores the non-zero values to save memory. We need to convert it into a **dense** matrix to view it effectively.
    # Make a DataFrame out of it
    vectorized_Xtrain = pd.DataFrame(vectorized_Xtrain.todense(), columns=vectorizer.get_feature_names(), index=Xtrain.index)
    # **A downside of the Count Vectorizer is that the uniqueness of words is not taken into consideration. This is where TF-IDF comes in.**
    # ---
    # Instantiate it
    transformer = TfidfTransformer()
    # Fit it
    transformer.fit(vectorized_Xtrain)
    # Transform it
    transformed_Xtrain = transformer.transform(vectorized_Xtrain)
    # Make a DataFrame out of it
    transformed_Xtrain = pd.DataFrame(transformed_Xtrain.todense(), 
                                columns=vectorized_Xtrain.columns, 
                                index=vectorized_Xtrain.index)
    # Fit the models
    ytrain.reset_index(drop=True, inplace=True)
    ytrain.head()
    transformed_Xtrain.reset_index(drop=True, inplace=True)
    transformed_Xtrain.head()
    # Model - Logistic Regression
    model = LogisticRegression(max_iter=10000)
    model.fit(transformed_Xtrain, ytrain)
    print("Train accuracy of Logistic Regression:")
    print(model.score(transformed_Xtrain, ytrain))
    # Alternative Model - Random Forest
    m = RandomForestClassifier(max_depth=8, n_estimators=500, random_state=166)   # these hyperparameters seem to be beneficial for the accuracy scores
    m.fit(transformed_Xtrain, ytrain)
    print("Train accuracy of Random Forest:")
    print(round(m.score(transformed_Xtrain, ytrain), 3))
    vectorized_Xtest=vectorizer.transform(Xtest)

    vectorized_Xtest.todense()
    vectorized_Xtest = pd.DataFrame(vectorized_Xtest.todense(), columns=vectorizer.get_feature_names(), index=Xtest.index)
    vectorized_Xtest.head()

    transformed_Xtest = transformer.transform(vectorized_Xtest)
    transformed_Xtest = pd.DataFrame(transformed_Xtest.todense(), 
                                columns=vectorized_Xtest.columns, 
                                index=vectorized_Xtest.index)
    transformed_Xtest.head()

    print("Test accuracy of Logistic Regression:")
    print(round(model.score(transformed_Xtest, ytest), 3))
    print("Test accuracy of Random Forest:")
    print(round(m.score(transformed_Xtest, ytest), 3))
    
    accuracy = cross_val_score(model, transformed_Xtrain, ytrain, cv=5, scoring='accuracy')
    print("model training cross-validation scores, log. reg. ", accuracy)  # mean score around 0,94

    accuracy = cross_val_score(m, transformed_Xtrain, ytrain, cv=5, scoring='accuracy')
    print("model training cross-validation scores, random forest ", accuracy)  # mean score around 0,94

    """
    print("Artist Prediction - Logistic Regression:")
    prediction = model.predict(transformed_Xtest)
    print(prediction)

    print("Artist Prediction - Random Forest:")
    prediction_forest = m.predict(transformed_Xtest)
    print(prediction_forest)

    print("Actual Artists:")
    print(ytest)
    """
    prediction = model.predict(transformed_Xtest)
    prediction_forest = m.predict(transformed_Xtest)

    ## Comparison Dataframe:
    comparedf = pd.DataFrame({"log. reg.":prediction,"rand. forest":prediction_forest,"data":ytest})
    print(comparedf.head(50))

    #vals = input('Enter S to save revised Bag Of Words: ')                # use this in the end!
    vals = 's'
    if (vals.lower() == 's'):     ### Save complete (train + test) Vectorized BOW Corpus
        vectorizer.fit(X)
        df_vectorized_X = vectorizer.transform(X)
        df_vectorized_X = pd.DataFrame(df_vectorized_X.todense(), columns=vectorizer.get_feature_names(), index=X.index)
        y.reset_index(inplace=True, drop=True)
        df_vectorized_X['000']=y
        df_vectorized_X.to_csv('bagoflyrics.csv', index=False)

    filename = 'lyrics_model.sav'                                 # saving logistic regression model
    pickle.dump(model, open(filename, 'wb'))    
    
if lyrics_file != None:
    # Load the dataset - this is the vectorized corpus prior to applying the Tf-Idf Transformer
    df_vectorized_corpus = pd.read_csv('bagoflyrics.csv')
    X = df_vectorized_corpus.drop(columns='000')
    y = df_vectorized_corpus['000']
#    print(y)
    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    
    f = open(lyrics_file, "r")
    newlyrics = f.read()
    
#    print (newlyrics)

#    print("************* CLEANED: ********************")
    newlyrics = clean_text(newlyrics, model=nlp)
#    print(newlyrics)
    df_newlyrics = pd.DataFrame({"lyrics":newlyrics}, index={-1})
    df_newlyrics= df_newlyrics['lyrics']
    vectorizer.fit(df_newlyrics)
    vectorized_newlyrics = vectorizer.transform(df_newlyrics)    
    vectorized_newlyrics = pd.DataFrame(vectorized_newlyrics.todense(), columns=vectorizer.get_feature_names(), index=df_newlyrics.index)
    
#    print(vectorized_newlyrics)
    
    X = X.append(vectorized_newlyrics, sort=False)

#    X = pd.concat([X,vectorized_newlyrics],join_axes=[X.columns])

#    print(X)
#    X = pd.concat([X,vectorized_newlyrics],axis=0).reindex(X.columns)
 #   X = pd.concat([X,vectorized_newlyrics],join_axes=[X.columns]).reindex(X.columns)
#    X = X.append(vectorized_newlyrics, sort=False).reindex(X.columns)
#    print(X)
    X.fillna(0, inplace = True)
#    print(X)

    transformer = TfidfTransformer()
    transformer.fit(X)
    transformed_X = transformer.transform(X)    
    transformed_X = pd.DataFrame(transformed_X.todense(), 
                     columns=X.columns, 
                     index=X.index)
#    print(transformed_X)

    model = LogisticRegression()
    modeltrainX=transformed_X[:-1]
    model.fit(modeltrainX, y)                           # re-train the model with extended vector dataframe

#    model = pickle.load(open("lyrics_model.sav", 'rb'))          # load saved model from disk
   
    prediction = model.predict(transformed_X)
    print('Text matched with the following artist: ')
    print(prediction[len(prediction)-1])                                   # last element will be the predicted artist
