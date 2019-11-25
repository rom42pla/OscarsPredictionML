#!python3
# -*- coding: utf-8 -*-

import pandas
import os
import ast
import numpy

import utils

def parseFileToDataframe(filepath, printDataframe = False, clean = True):
    # starts the timers
    sectionTimer = utils.Timer()
    print(f"Parsing file {filepath} ({'{0:.1f}'.format(os.path.getsize(filepath) * 2**(-20))}Mb)...")
    sectionTimer.restartTimer()

    # parses every file's line
    data = []
    with open(filepath, mode="r") as fp:
        data = fp.readlines()
    
    def splitLines(line):
        # sets file's separator
        fieldSeparator = " "
        extension = filepath[-4:]
        if(extension == ".csv"):
            fieldSeparator = ", "
        elif(extension == ".tsv"):
            fieldSeparator = "\t"
        # splits the line
        splittedLine = line.split(fieldSeparator)
        return splittedLine

    def cleanErrors(line):
        cleanLine = []
        for i, string in enumerate(line):
            # strips and lower the string
            cleanString = string.strip().lower()
            # if the current column is "ID"
            if(i == 0):
                tmp = ""
                for c in cleanString:
                    if(c.isdigit()):
                        tmp += c
                    else:
                        tmp += str(ord(c))
                cleanString = int(tmp)
            else:
                # eventually parses the "string number" to a number
                try:
                    n = float(cleanString)
                    cleanString = n
                except:
                    if(len(cleanString) > 0):
                        # checks null
                        if(cleanString == "null"):
                            cleanString = numpy.nan
                        # checks for booleans
                        elif(cleanString == "true"):
                            cleanString = 1
                        elif(cleanString == "false"):
                            cleanString = 0
                        # checks for lists
                        elif(cleanString[0] == "["):
                            cleanString = ast.literal_eval(cleanString)
                        # checks some column-related errors
                        elif(i == 5 and cleanString in {"unrated", "not rated"}):
                            cleanString = None
            # updates the line
            cleanLine += [cleanString]
        return cleanLine

    # transforms parsed liness into strings
    lines = list(map(splitLines, data))
    features, data = list(map(lambda x: x.strip(), lines[0])), lines[1:]
    # checks if a correction has to be made
    if(cleanErrors):
        data = list(map(cleanErrors, data))
    # transforms the list to a dataframe
    dataframe = pandas.DataFrame(data=data, columns=features, dtype=object)
    if(printDataframe):
        print(dataframe)

    print(f"\t...parsed a dataframe of {dataframe.shape[0]} movies and {dataframe.shape[1]} features in {sectionTimer.getHumanReadableElapsedTime()}")
    return dataframe