import time, re

class Timer:
    def __init__(self):
        self.time = None

    def restartTimer(self):
        self.time = time.time()

    def getElapsedTime(self):
        return time.time() - self.time

    def getHumanReadableElapsedTime(self):
        elapsedTime = self.getElapsedTime()
        if(elapsedTime < 120):
            return str('%.2f' % (time.time() - self.time)) + " seconds"
        else:
            return str('%.2f' % ((time.time() - self.time) / 60)) + " minutes"

def getXandy(df):
    return df.loc[:, :"otherPrizes"].to_numpy(), df.loc[:, "oscar"].to_numpy()
    
def matchRE(regex, string):
    return re.compile(regex).match(string) != None

def toCamelcase(string):
    res = ""
    for i, word in enumerate(string.lower().split()):
        if i == 0:
            res += word.strip()
        else:
            res += word.capitalize().strip()
    return res