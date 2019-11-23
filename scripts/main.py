import sys
import os
import pandas

import utils
import parsing

if __name__ == "__main__":
    totalTimer = utils.Timer()
    totalTimer.restartTimer()

    consoleArgs = sys.argv[1:]
    # if no arguments are given, loads the first file in the folder
    path = "./data"
    if(len(consoleArgs) == 0):
        files = [f for f in os.listdir(path) if os.path.isfile(path + "/" + f)]
        print(files)
        dataFilepath = path + "/" + files[0]
    # else loads the specified file
    else:
        dataFilepath = consoleArgs[0]
    # parses the file to a dataframe
    dataframe = parsing.parseFileToDataframe(filepath=dataFilepath, printDataframe=True)

    print(f"\nEverything done in {totalTimer.getHumanReadableElapsedTime()}")