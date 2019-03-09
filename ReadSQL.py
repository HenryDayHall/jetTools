import sqlite3
import numpy as np

def readSelected(databaseName, selectedFields):
    connection = sqlite3.connect(databaseName)
    cursor = connection.cursor()
    sql = "SELECT " + ", ".join(selectedFields) + " FROM GenParticles"
    cursor.execute(sql)
    out = cursor.fetchall()
    connection.close()
    return out

def readAll(databaseName):
    out = readSelected(databaseName, '*')
    return out


def makeCheckfile(databaseName):
    fields = ["MCPID", "Status", "IsPU", "Charge", "Mass", "E", "Px", "Py", "Pz", "P", "PT", "Eta", "Phi", "Rapidity", "CtgTheta", "D0", "DZ", "T", "X", "Y", "Z"]
    out = readSelected(databaseName, fields)
    cppOut = [', '.join((cppLike(x) for x in line)) for line in out]
    testName = databaseName.split('.')[0] + "_python.txt"
    with open(testName, 'w') as outFile:
        for line in cppOut:
            outFile.write(str(line) + "\n")
    print(f"Written to {testName}")

def cppLike(x):
    if x == 0:
        x = abs(x)
    if x == int(x):
        return str(int(x))
    else:
        return str(x)


def main():
    makeCheckfile("/home/henry/lazy/tag_1_delphes_events.db")

if __name__ == '__main__':
    main()
