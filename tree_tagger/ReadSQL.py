import sqlite3
from tree_tagger import Components
from tree_tagger import ReadHepmc
from ipdb import set_trace as st
import numpy as np


def read_selected(databaseName, selectedFields, tableName="GenParticles", where=None, field_in_list=None):
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
    # if there are conditonals add them here
    where_components = []
    if where is not None:
        where_components.append(where)
    if field_in_list is not None:
        column = field_in_list[0]
        in_list = ', '.join(field_in_list[1])
        where_components.append(column + " IN (" + in_list + ")")
    if len(where_components) > 0:
        sql += " WHERE " + ' AND '.join(where_components)
    # now execute
    cursor.execute(sql)
    out = cursor.fetchall()
    connection.close()
    out = np.array([list(row) for row in out])
    return out


def read_all(databaseName, tableName="GenParticles"):
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
    out = read_selected(databaseName, '*', tableName)
    return out


def trackTowerCreators(databaseName, fields, event_condition):
    creators = []
    trackParticleIDs = read_selected(databaseName, ["Particle"], tableName="Tracks", where=event_condition)
    trackParticleIDs = [str(p[0]) for p in trackParticleIDs]
    trackParticles = read_selected(databaseName, selectedFields=fields, field_in_list=("ID", trackParticleIDs), where=event_condition)
    trackParticles = np.hstack((np.zeros((trackParticles.shape[0], 1)), trackParticles))
    towerParticleIDs = read_selected(databaseName, ["Particle"], tableName="TowerLinks", where=event_condition)
    towerParticleIDs = [str(p[0]) for p in towerParticleIDs if str(p[0]) not in trackParticleIDs]
    towerParticles = read_selected(databaseName, selectedFields=fields, field_in_list=("ID", towerParticleIDs), where=event_condition)
    towerParticles = np.hstack((np.ones((towerParticles.shape[0], 1)), towerParticles))
    theCreators = np.vstack((trackParticles, towerParticles))
    return theCreators


def trackTowerDict(databaseName, event_condition):
    tracksList = read_selected(databaseName, ["ID", "Particle"], tableName="Tracks", where=event_condition)
    trackDict = {int(tID) : int(pID) for (tID, pID) in tracksList}
    towerList = read_selected(databaseName, ["ID", "Particle"], tableName="TowerLinks", where=event_condition)
    towerDict = {int(tID) : int(pID) for (tID, pID) in towerList}
    return trackDict, towerDict


def read_tracks_towers(particle_collection, database_name, event_n):
    event_condition = f"EVENT = {event_n}"
    # get the sql keys and check the pids match
    particle_data = read_selected(database_name, ['ID', 'MCPID'], "GenParticles", where=event_condition)
    assert len(particle_data) == len(particle_collection), f"Difernet number of particles in .db ({len(particle_data)} and .hepmc ({len(particle_collection)})."
    sql_pids = np.array([int(line[1]) for line in particle_data], dtype=int)
    np.testing.assert_array_equal(sql_pids, particle_collection.pids, 
                                  "Particle pids dont match")
    sql_particle_keys = np.array([int(line[0]) for line in particle_data], dtype=int)
    sql_pkeys_list = sql_particle_keys.tolist()
    for particle_n, key in enumerate(sql_particle_keys):
        particle_collection.particle_list[particle_n].sql_key = key
    particle_collection.updateParticles()
    np.testing.assert_array_equal(sql_particle_keys, particle_collection.sql_keys,
                                  "Error setting sql keys in the particle collection")
    # make a list of tracks
    track_list = []
    track_data = read_selected(database_name, ['ID', 'MCPID', 'Particle', 'Charge', 'P', 'PT',
                                              'Eta', 'Phi', 'CtgTheta', 'EtaOuter', 'PhiOuter',
                                              'T', 'X', 'Y', 'Z', 'Xd', 'Yd', 'Zd', 'L', 'D0',
                                              'DZ', 'TOuter', 'XOuter', 'YOuter', 'ZOuter'],
                                "Tracks", where=event_condition)
    track_id = 0
    for line in track_data:
        pkey = int(line[2])
        p_idx = np.where(particle_collection.sql_keys==pkey)[0][0]
        particle = particle_collection.particle_list[p_idx]
        track = Components.MyTrack(global_track_id=track_id,
                                   pid=int(line[1]),
                                   sql_key=int(line[0]),
                                   particle_sql_key=pkey,
                                   particle=particle,
                                   charge=float(line[3]),
                                   p=float(line[4]),
                                   pT=float(line[5]),
                                   eta=float(line[6]),
                                   phi=float(line[7]),
                                   ctgTheta=float(line[8]),
                                   etaOuter=float(line[9]),
                                   phiOuter=float(line[10]),
                                   t=float(line[11]),
                                   x=float(line[12]),
                                   y=float(line[13]),
                                   z=float(line[14]),
                                   xd=float(line[15]),
                                   yd=float(line[16]),
                                   zd=float(line[17]),
                                   l=float(line[18]),
                                   d0=float(line[19]),
                                   dZ=float(line[20]),
                                   t_outer=float(line[21]),
                                   x_outer=float(line[22]),
                                   y_outer=float(line[23]),
                                   z_outer=float(line[24]))
        track_id += 1
        track_list.append(track)
    # make list of towers
    tower_list = []
    tower_data = read_selected(database_name, ['ID', 'T', 'NTimeHits', 'Eem', 'Ehad',
                                              'Edge1', 'Edge2', 'Edge3', 'Edge4',
                                              'E', 'ET', 'Eta', 'Phi'],
                               "Towers", where=event_condition)
    link_data = read_selected(database_name, ['Tower', 'Particle'], "TowerLinks",
                              where=event_condition)
    link_data = np.array(link_data, dtype=int)
    tower_id = 0
    for line in tower_data:
        tkey = int(line[0])
        pkeys = [link[1] for link in link_data if link[0] == tkey]
        p_idxs = np.where(np.isin(particle_collection.sql_keys, pkeys))[0]
        particles = [particle_collection.particle_list[i] for i in p_idxs]
        tower = Components.MyTower(global_tower_id=tower_id,
                                   pids=[p.pid for p in particles],
                                   sql_key=int(line[0]),
                                   particles_sql_keys=pkeys,
                                   particles=particles,
                                   t=float(line[1]),
                                   nTimesHits=int(line[2]),
                                   eem=float(line[3]),
                                   ehad=float(line[4]),
                                   edges=[float(edge) for edge in line[5:9]],
                                   e=float(line[9]),
                                   et=float(line[10]),
                                   eta=float(line[11]),
                                   phi=float(line[12]))
        tower_list.append(tower)
        tower_id += 1
    return track_list, tower_list


def main(event_num=0):
    """ """
    hepmc_name = "/home/henry/lazy/dataset2/h1bBatch2.hepmc"
    database_name = "/home/henry/lazy/h1bBatch2.db"
    event = ReadHepmc.read_file(hepmc_name, event_num, event_num+1)[0]
    track_list, tower_list = read_tracks_towers(event, database_name, event_num)
    observations = Components.Observables(tracks=track_list, towers=tower_list)
    return event, track_list, tower_list, observations

if __name__ == '__main__':
    main()
