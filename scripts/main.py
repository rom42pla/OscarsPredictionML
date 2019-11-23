import pandas

import utils
import parsing

if __name__ == "__main__":
    totalTimer = utils.Timer()
    totalTimer.restartTimer()

    # parses the file into a dataframe
    dataFilepath = "./data/sample.tsv"
    dataframe = parsing.parseFileToDataframe(filepath=dataFilepath, printDataframe=True)

    print(f"\nEverything done in {totalTimer.getHumanReadableElapsedTime()}")