import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import utils

'''
def plotBarChart(x, y, xLabel=None, yLabel=None, textSize=24, title="Sample bar plot"):
    figure(num=None, figsize=(24, 8), facecolor='w', edgecolor='k', title=title)
    plt.tight_layout()
    xTicks = np.arange(len(x))
    plt.bar(x=xTicks, height=y, align="center", alpha=0.25)
    plt.xticks(xTicks, x)
    if(xLabel != None):
        plt.xlabel(xLabel, fontsize=textSize)
    if(yLabel != None):
        plt.ylabel(yLabel, fontsize=textSize)
    plt.title = title
    fileName = "../imgs/" + utils.toCamelcase(title) + ".png"
    plt.savefig(fname=fileName)
'''

def plotBarChart(x, y, xLabel=None, yLabel=None, textSize=24, title="Sample bar plot"):
    fig = figure(num=None, figsize=(24, 8), facecolor='w', edgecolor='k')
    fig.tight_layout()
    xTicks = np.arange(len(x))
    plt.bar(x=xTicks, height=y, align="center", alpha=0.25)
    plt.xticks(xTicks, x)
    if(xLabel != None):
        plt.xlabel(xLabel, fontsize=textSize)
    if(yLabel != None):
        plt.ylabel(yLabel, fontsize=textSize)
    plt.title(title)
    fileName = "../imgs/" + utils.toCamelcase(title) + ".png"
    plt.savefig(fname=fileName)
    
def plotPieChart(x, labels, textSize=24, title="Sample pie plot"):
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    figure(num=None, figsize=(12, 8), facecolor='w', edgecolor='k')
    plt.tight_layout()
    plt.pie(x=x, labels=labels, autopct=make_autopct(x))
    plt.title(title)
    fileName = "../imgs/" + utils.toCamelcase(title) + ".png"
    plt.savefig(fname=fileName)