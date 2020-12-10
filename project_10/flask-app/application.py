#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:38:00 2020

@author: boti
"""

from flask import Flask, render_template, request
from recommender_tools2 import movie_recommender
from recommender_tools2 import read_matrix
from recommender_tools2 import find_movie_title
from threading import Thread
import logging
import numpy as np
import time

app = Flask(__name__)

@app.route('/index')
@app.route('/')
def index():
    return render_template("index.html", btn_state = "disabled")

@app.route('/recommend', methods=['GET','POST'])
def recommend():
    if request.method == 'GET':
        while thread.isAlive():
            time.sleep(1)
        usrinput = dict(request.args)            
        logging.critical(usrinput)
        moviedata = []                                   # first find the matching movies in the movie db:
        moviedata.extend([find_movie_title(usrinput['movie1'],usrinput['year1']),find_movie_title(usrinput['movie2'],usrinput['year2']),find_movie_title(usrinput['movie3'],usrinput['year3'])])
        titles = [row[0] for row in moviedata]
        years = [row[1] for row in moviedata]        
        form_filled = 1
        for element in titles:
            form_filled *= len(element)
        if form_filled == 0:                        # if form incomplete:
            return render_template("index.html", movie1 = titles[0], movie2 = titles[1], movie3 = titles[2], year1 = years[0], year2 = years[1], year3 = years[2], btn_state = "disabled", warning_message = "Please provide three movies...")
        elif len(titles) != len(set(titles)):       # checks duplicates among the usr movies
            return render_template("index.html", movie1 = titles[0], movie2 = titles[1], movie3 = titles[2], year1 = years[0], year2 = years[1], year3 = years[2], btn_state = "disabled", warning_message = "Duplicate movies found...")
        elif 'submit' in usrinput:                  # valid submission: generate results and open recommendations.html
            result = movie_recommender(titles, years)
            for i in range(len(titles)):
                titles[i] = titles[i] + " (" + str(years[i]) + ")"    # submit results and original movies + years
            return render_template("recommendations.html", originals = titles, results = result)
        else:                # displays validated movie titles on index.html
            return render_template("index.html", movie1 = titles[0], movie2 = titles[1], movie3 = titles[2], year1 = years[0], year2 = years[1], year3 = years[2], btn_state = "", warning_message = "")

    
if __name__ == '__main__':
    global thread
    thread = Thread(target = read_matrix)
    thread.start()
    app.run(debug=True, port=5000)