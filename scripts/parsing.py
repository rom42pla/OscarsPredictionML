#!python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import ast
import numpy as np

import utils
from utils import matchRE

def cleanErrors(line):
    cleanLine = []
    for i, string in enumerate(line):
        # strips and lower the string
        cleanString = str(string).strip().lower()
        # id: if the current column is "ID"
        if(i == 0):
            tmp = ""
            for c in cleanString:
                if(c.isdigit()):
                    tmp += c
                else:
                    tmp += str(ord(c))
            cleanString = int(tmp)
        # "title", "storyline" and "director" should be strings because there is just one of them for each movie, so it's fine
        elif(i in {1, 2, 7}):
            if(cleanString in {"nan", "null", "none"}):
                cleanString = np.nan
        # classification: if the current column is "classification", there are two types of it that are in fact the same
        elif(i == 5):
            if(cleanString in {"unrated", "not rated"}):
                cleanString = np.nan
        # numbers: if the current column is "year", "runningTime", "budget", "income", "imdbReviews", "imdbRating", "metascoreReviews", "metascoreRating"
        elif(i in {3, 4, 11, 12, 13, 14, 15, 16}):
            try:
                n = float(cleanString)
                cleanString = n
            except ValueError:
                cleanString = np.nan
        # other cases
        else:
            if(len(cleanString) > 0):
                # checks null
                if(cleanString in {"nan", "null", "none"} or cleanString == "[]"):
                    cleanString = np.nan
                # checks for booleans
                elif(cleanString == "true"):
                    cleanString = 1
                elif(cleanString == "false"):
                    cleanString = 0
                # checks for lists
                elif(matchRE('\[".*"\]', cleanString) or matchRE("\['.*'\]", cleanString)):
                    cleanString = ast.literal_eval(cleanString)
                else:
                    cleanString = np.nan
            else:
                cleanString = np.nan
        # updates the line
        cleanLine += [cleanString]
    return cleanLine

def parseFileToDataframe(filepath, printDataframe = False, clean = True, rows = None):
    # starts the timers
    sectionTimer = utils.Timer()
    print(f"Parsing file {filepath} ({'{0:.1f}'.format(os.path.getsize(filepath) * 2**(-20))}Mb)...")
    sectionTimer.restartTimer()

    extension = filepath[-4:]
    if(extension == ".csv"):
        separator = ","
    elif(extension == ".tsv"):
        separator = "\t"
    else:
        separator = " "
    df = pd.read_csv(filepath, nrows=rows, sep=separator, na_values=["null"], index_col=False, encoding="utf_8", dtype= str, header=0)

    # transforms parsed liness into strings
    features, df_list = list(df.columns), df.values.tolist()
    # checks if a correction has to be made
    if(clean):
        df_list = list(map(cleanErrors, df_list))
    # transforms the list back to a dataframe
    df = pd.DataFrame(data=df_list, columns=features, dtype=object)
    if(printDataframe):
        print(df)
    print(f"\t...parsed a dataframe of {df.shape[0]} movies and {df.shape[1]} features in {sectionTimer.getHumanReadableElapsedTime()}")
    return df

def writeDataframeToFile(df, filepath):
    df.to_csv(path_or_buf=filepath, index=False, sep="\t")