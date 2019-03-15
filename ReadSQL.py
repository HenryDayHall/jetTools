import sqlite3
import numpy as np

def readSelected(databaseName, selectedFields, tableName="GenParticles"):
    """
    Read all entries of names columns from table in database

    Parameters
    ----------
    databaseName : string
        file name of the database to be read

    selectedFields : list of strings
        name sof the fields to be read in order
        
    tableName : string
        name of the table to read from

    Returns
    -------
    out : list of tuples
        list where each item is a record.
        the record is a tuple with the table contents in the requested columns

    """
    connection = sqlite3.connect(databaseName)
    cursor = connection.cursor()
    sql = "SELECT " + ", ".join(selectedFields) + " FROM " + tableName
    cursor.execute(sql)
    out = cursor.fetchall()
    connection.close()
    return out

def readAll(databaseName, tableName="GenParticles"):
    """
    REad all the data from a table in a database

    Parameters
    ----------
    databaseName : string
        file name of the database to be read

    tableName : string
        name of the table to read from
        

    Returns
    -------
    out : list of tuples
        list where each item is a record.

    """
    out = readSelected(databaseName, '*', tableName)
    return out


def makeCheckfile(databaseName):
    """
    Pakes a csv file of all the data in the database,
    with format as close as possible to the c++ output.
    

    Parameters
    ----------
    databaseName : string
        file name os the database to be read.

    """
    fields = ["MCPID", "Status", "IsPU", "Charge", "Mass", "E", "Px", "Py", "Pz", "P", "PT", "Eta", "Phi", "Rapidity", "CtgTheta", "D0", "DZ", "T", "X", "Y", "Z"]
    out = readSelected(databaseName, fields)
    cppOut = [', '.join((cppLike(x) for x in line)) for line in out]
    testName = databaseName.split('.')[0] + "_python.txt"
    with open(testName, 'w') as outFile:
        for line in cppOut:
            outFile.write(str(line) + "\n")
    print(f"Written to {testName}")

def cppLike(x):
    """
    Format a number to be closer to the format of a c++ print
    

    Parameters
    ----------
    x : float or int
        number to be printed
        

    Returns
    -------
    : str
        the c++ like string format

    """
    if x == 0:
        x = abs(x)
    if x == int(x):
        return str(int(x))
    else:
        return str(x)


def checkReflection(databaseName=None, **kwargs):
    """
    Check for agreement between parent and child fields

    Parameters
    ----------
    databaseName : str
        path and file name of the database to be read
        

    """
    if databaseName is not None:
        fields = ["ID", "M1", "M2", "D1", "D2"]
        fromDatabase = readSelected(databaseName, fields)
        IDs = np.array([d[0] for d in fromDatabase])
        parents = np.array([d[1:3] for d in fromDatabase])
        children = np.array([d[3:5] for d in fromDatabase])
    else:
        IDs = kwargs.get('IDs')
        parents = kwargs.get('parents')
        children = kwargs.get('children')
    listIDs = list(IDs)
    for row, this_id in enumerate(IDs):
        for p in parents[row]:
            if p is not None:
                assert p in IDs, f"The parent {p} of {this_id} is invalid"
                p_row = listIDs.index(p)  # find the row of the parent
                assert this_id in children[p_row], f"Parent {p} of {this_id} not acknowledging child"
        for c in children[row]:
            if c is not None:
                assert c in IDs, f"The child {c} of {this_id} is invalid"
                c_row = listIDs.index(c)  # find the row of the child
                assert this_id in children[c_row], f"Child {c} of {this_id} not acknowledging parent"

def checkPIDMatch(databaseName, table1, refField, table2, PIDfield="MCPID"):
    # in table 1 we need the ref field and the pid field
    out1 = readSelected(databaseName, [refField, PIDfield], table1)
    # in table2 we need the ID field and the pid field
    out2 = readSelected(databaseName, ["ID", PIDfield], table2)
    # convert the second table to a dict
    out2 = dict(out2)
    for foreignKey, PID in out1:
        assert PID == out2[foreignKey], f"Error, track PID mismatch for particle {foreignKey}"


def main():
    """ """
    databaseName = "/home/henry/lazy/tag_1_delphes_events.db"
    #makeCheckfile(databaseName)
    #checkReflection(databaseName)
    checkPIDMatch(databaseName, "Tracks", "Particle", "GenParticles")

if __name__ == '__main__':
    #main()
    pass
