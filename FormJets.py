import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from ReadSQL import readSelected
from ReadHepmc import Hepmc_event
from ipdb import set_trace as st
import DrawTrees

class JetCluster:
    def __init__(self):
        self.obsIDs = None
        self.trackTowerInner = None
        self.clusterIDs = None
        self.cluster_daughterss = None
        self.cluster_mothers = None
        self.pts = None
        self.etas = None
        self.phis = None
        self.energies = None
        self.ranks = None
        self.hasObs = False

    def __len__(self):
        return len(self.clusterIDs)

    def addObsLevel(self):
        if self.hasObs:
            return
        else:
            self.hasObs = True # check we don't do this twice
        self.obsParticleIDs = np.zeros_like(self.obsIDs)
        trackMask = self.trackTowerInner == 0
        towerMask = self.trackTowerInner == 1
        # would need to do something here to plit towers intoo particles if the real towers where used
        obsIndices = np.where(self.obsIDs > 0)[0]
        # self.obsParticleIDs[trackMask] = np.vectorize(trackParticleDict.__getitem__)(self.obsIDs[trackMask])
        # self.obsParticleIDs[towerMask] = np.vectorize(trackParticleDict.__getitem__)(self.obsIDs[trackMask])
        self.obsParticleIDs = self.obsIDs[obsIndices]
        # give these cluster IDs
        startID = max(self.clusterIDs) + 1
        obsClusterIDs = startID + np.arange(len(self.obsParticleIDs))
        self.clusterIDs = np.append(self.clusterIDs, obsClusterIDs)
        self.cluster_mothers = np.append(self.cluster_mothers, self.clusterIDs[obsIndices])
        # the observed particles wont have and daughters themselves
        self.cluster_daughterss += [[] for _ in obsIndices]
        # also we want the obs to eb the daughters of the relevant tracks
        for index, cID in zip(obsIndices, obsClusterIDs):
            self.cluster_daughterss[index].append(cID)
        assert len(self.cluster_daughterss) == len(self.cluster_mothers)

    def addGlobalID(self, maxIDOutside):
        # fit with particleIDs
        lowest = max(maxIDOutside, np.max(self.obsParticleIDs)) + 1
        # give all the particles sequntial IDs
        self.IDs = np.arange(len(self.clusterIDs)) + lowest
        # store the observations at the end of this array
        self.IDs[-len(self.obsParticleIDs):] = self.obsParticleIDs
        # need to add ranks for the observations
        self.ranks += 1-min(self.ranks)  #enusre the ranks start at 1
        self.ranks = np.append(self.ranks, np.zeros_like(self.obsParticleIDs))
        # now we want to create a self.mothers and self.daughters that refernces this
        self.mothers = []
        self.daughters = []
        for cluster_mother, cluster_daughters in zip(self.cluster_mothers, self.cluster_daughterss):
            if cluster_mother > 0:  # mother in jet
                motherIndices = np.where(self.clusterIDs == cluster_mother)[0]
                assert len(motherIndices) == 1
                self.mothers.append(self.IDs[motherIndices[0]])
            else:  # root particle
                self.mothers.append(-1)
            daughterIDs = [self.IDs[np.where(self.clusterIDs == d)] for d in cluster_daughters]
            self.daughters.append(daughterIDs)
        assert len(self.ranks) == len(self.IDs)
        assert len(self.mothers) == len(self.IDs)
        assert len(self.daughters) == len(self.IDs)
        

class ClusterForest:
    def __init__(self, trackList, deltaR=.4, exponent_multiplyer=-1):
        self.trackList = trackList
        self.deltaR = deltaR
        self.exponent = 2 * exponent_multiplyer
        # each cluster is a dictionary with referance to it's daughters and it's parameters
        # the tracks make the intial clusters
        self.clusters = [{'pt':t[2], 'eta':t[3], 'phi':t[4], 'energy': t[5],
                          'obsID': int(t[1]),  # will be None for all other clustered objects
                          'trackTowerInner': t[0], # 0=track, 1=tower, 2=inner
                          'clusterID': ID,  # each cluster will have a unique id
                          'daughters': [], 'mother': -1,  # -1 is reserved as "No ID"
                          'rank': 0}
                         for ID, t in enumerate(trackList)]
        # keep track of which clusters don't yet have a mother
        self.currently_avalible = len(self.clusters)
        self._calculateDistances()
        # as we go note the root notes of the clusters
        self.root_clusterIDs = []

    def sort_clusters(self):
        self.sorted_clusters = sorted(self.clusters, key=lambda c:c['clusterID'])
        self.grouped_clusters = [self.getDecendants(lastOnly=False, clusterID=clusterID)
                                 for clusterID in self.root_clusterIDs]
        self.JetList = []
        for group in self.grouped_clusters:
            jet = JetCluster()
            jet.obsIDs = np.array([p['obsID'] for p in group])
            jet.trackTowerInner = np.array([p['trackTowerInner'] for p in group])
            jet.clusterIDs = np.array([p['clusterID'] for p in group])
            jet.cluster_daughterss = [p['daughters'] for p in group]
            jet.cluster_mothers = np.array([p['mother'] for p in group])
            jet.pts = np.array([p['pt'] for p in group])
            jet.etas = np.array([p['eta'] for p in group])
            jet.phis = np.array([p['phi'] for p in group])
            jet.energies = np.array([p['energy'] for p in group])
            jet.ranks = np.array([p['rank'] for p in group])
            self.JetList.append(jet)
        return self.JetList

    
    def _calculateDistances(self):
        # this is caluculating all the distances
        self.distances = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                elif column == row:
                    distance = self.clusters[row]['pt']**self.exponent * self.deltaR
                else:
                    angular_diffrence = self.clusters[row]['phi'] - self.clusters[column]['phi']
                    angular_distance = min(abs(angular_diffrence), abs(2*np.pi - angular_diffrence))
                    distance = min(self.clusters[row]['pt']**self.exponent, self.clusters[column]['pt']**self.exponent) *\
                               ((self.clusters[row]['eta'] - self.clusters[column]['eta'])**2 +
                               (angular_distance)**2)
                self.distances[row, column] = distance

    
    def _recalculateOne(self, cluster_num):
        for row in range(self.currently_avalible):
            column = cluster_num
            if column > row:
                row, column = column, row  # keep the upper triangular form
            if column == row:
                distance = self.clusters[row]['pt']**self.exponent * self.deltaR
            else:
                angular_diffrence = self.clusters[row]['phi'] - self.clusters[column]['phi']
                angular_distance = min(abs(angular_diffrence), abs(2*np.pi - angular_diffrence))
                distance = min(self.clusters[row]['pt']**self.exponent, self.clusters[column]['pt']**self.exponent) *\
                           ((self.clusters[row]['eta'] - self.clusters[column]['eta'])**2 +
                           (angular_distance)**2)
            self.distances[row, column] = distance

    def _merge_clusters(self, cluster_index1, cluster_index2):
        new_cluster = self._combine(cluster_index1, cluster_index2)
        # move the first cluster to the back without replacement
        cluster1 = self.clusters.pop(cluster_index1)
        self.clusters.append(cluster1)
        # move the second cluster to the back but replace it with the new cluster
        cluster2 = self.clusters[cluster_index2]
        self.clusters.append(cluster2)
        self.clusters[cluster_index2] = new_cluster
        # one less cluster avalible
        self.currently_avalible -= 1
        
        # delete the first row and column of the merge
        self.distances = np.delete(self.distances, (cluster_index1), axis=0)
        self.distances = np.delete(self.distances, (cluster_index1), axis=1)

        # now recalculate for the new cluster
        self._recalculateOne(cluster_index2)

    def _remove_cluster(self, cluster_index):
        # move the first cluster to the back without replacement
        cluster = self.clusters.pop(cluster_index)
        self.root_clusterIDs.append(cluster["clusterID"])
        self.clusters.append(cluster)
        # delete the row and column
        self.distances = np.delete(self.distances, (cluster_index), axis=0)
        self.distances = np.delete(self.distances, (cluster_index), axis=1)
        # one less cluster avalible
        self.currently_avalible -= 1
        

    def assignMothers(self):
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self.distances), self.distances.shape)
            if row == column:
                self._remove_cluster(row)
            else:
                self._merge_clusters(row, column)

    def pltAssignMothers(self):
        plt.axis([-5, 5, -np.pi-0.5, np.pi+0.5])
        etas = [c['eta'] for c in self.clusters]
        phis = [c['phi'] for c in self.clusters]
        energies = [c['energy']/5. for c in self.clusters]
        plt.scatter(etas, phis, energies, marker='D', c='w')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylabel(r"$\phi$ - barrel angle")
        plt.xlabel(r"$\eta$ - psudo rapidity")
        plt.title("Detected Hits")
        plt.gca().set_facecolor('gray')
        # for getting rid of the axis
        #plt.gca().get_xaxis().set_visible(False)
        #plt.gca().get_yaxis().set_visible(False)
        #plt.gca().spines['top'].set_visible(False)
        #plt.gca().spines['right'].set_visible(False)
        #plt.gca().spines['bottom'].set_visible(False)
        #plt.gca().spines['left'].set_visible(False)
        plt.pause(0.05)#
        input("Press enter to start clustering")
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self.distances), self.distances.shape)
            if row == column:
                decendents = self.getDecendants(lastOnly=True, cluster=self.clusters[row])
                etas = [d['eta'] for d in decendents]
                phis = [d['phi'] for d in decendents]
                plt.scatter(etas, phis, marker='x')
                print(f"Added jet of {len(decendents)} tracks, {self.currently_avalible} clusters unfinished")
                plt.pause(0.05)
                input("Press enter for next cluster")
                self._remove_cluster(row)
            else:
                self._merge_clusters(row, column)
        plt.show()

    def cluster_from_ID(self, clusterID):
        cluster = next((c for c in self.clusters if c['clusterID'] == clusterID),
                       None)
        if cluster is not None:
            return cluster
        raise ValueError(f"No cluster with ID {clusterID}")

    def getDecendants(self, lastOnly=True, cluster=None, clusterID=None):
        if clusterID is None and cluster is None:
            raise TypeError("Need to specify a cluster")
        elif cluster is None:
            cluster = self.cluster_from_ID(clusterID)
        decendents = []
        if not lastOnly:
            decendents.append(cluster)
        # bu this point we have the first cluster
        if len(cluster['daughters']) == 0:
            # just the one
            return [cluster]
        toCheck = [daughter for daughter in cluster['daughters']]
        while len(toCheck) > 0:
            cluster = self.cluster_from_ID(toCheck.pop())
            if len(cluster['daughters']) == 0 or not lastOnly:
                decendents.append(cluster)
            toCheck += cluster['daughters']
        return decendents
            

    def _combine(self, cluster_index1, cluster_index2):
        new_index = len(self.clusters)
        self.clusters[cluster_index1]['mother'] = new_index
        self.clusters[cluster_index2]['mother'] = new_index
        cluster1 = self.clusters[cluster_index1]
        cluster2 = self.clusters[cluster_index2]
        rank = max(cluster1['rank'], cluster2['rank']) + 1
        new_cluster = {'daughters': [cluster1['clusterID'], cluster2['clusterID']],
                'mother': -1, 'clusterID': len(self.clusters), 'obsID': -1,
                'trackTowerInner': 2, 'rank': rank}
        cluster1 = self.clusters[cluster_index1]
        cluster2 = self.clusters[cluster_index2]
        new_cluster['pt'] = cluster1['pt'] + cluster2['pt']
        new_cluster['energy'] = cluster1['energy'] + cluster2['energy']
        new_cluster['eta'] = (cluster1['pt']*cluster1['eta'] + cluster2['pt']*cluster2['eta'])/new_cluster['pt']
        new_cluster['phi'] = (cluster1['pt']*cluster1['phi'] + cluster2['pt']*cluster2['phi'])/new_cluster['pt']
        # one scheme (recombination schemes) look this up get fastjet manuel to see how it does for each algorithm
        return new_cluster


def test_ClusterTree():
    # equlatral triangle
    eta = [0, 1, 2]
    phi = [0, np.sqrt(3), 0]
    ids = [0, 1, 2]
    pts = [1, 1, 1]
    trackList = np.array([ids, pts, eta, phi]).T
    clust = ClusterForest(trackList)
    clust._calculateDistances()
    lower_left = clust.distances[-1, 0]
    diag = clust.distances[0, 0]
    for row in range(len(clust.distances)):
        for column in range(len(clust.distances)):
            if column > row:
                assert clust.distances[row, column] == np.inf
            elif column == row:
                np.testing.assert_allclose( clust.distances[row, column], diag)
            else:
                np.testing.assert_allclose( clust.distances[row, column], lower_left)
    

def trackTowerCreators(databaseName, fields):
    creators = []
    trackParticleIDs = readSelected(databaseName, ["Particle"], tableName="Tracks")
    trackParticleIDs = [str(p[0]) for p in trackParticleIDs]
    trackParticles = readSelected(databaseName, selectedFields=fields, field_in_list=("ID", trackParticleIDs))
    trackParticles = np.hstack((np.zeros((trackParticles.shape[0], 1)), trackParticles))
    towerParticleIDs = readSelected(databaseName, ["Particle"], tableName="TowerLinks")
    towerParticleIDs = [str(p[0]) for p in towerParticleIDs if str(p[0]) not in trackParticleIDs]
    towerParticles = readSelected(databaseName, selectedFields=fields, field_in_list=("ID", towerParticleIDs))
    towerParticles = np.hstack((np.ones((towerParticles.shape[0], 1)), towerParticles))
    theCreators = np.vstack((trackParticles, towerParticles))
    return theCreators

def trackTowerDict(databaseName):
    tracksList = readSelected(databaseName, ["ID", "Particle"], tableName="Tracks")
    trackDict = {int(tID) : int(pID) for (tID, pID) in tracksList}
    towerList = readSelected(databaseName, ["ID", "Particle"], tableName="TowerLinks")
    towerDict = {int(tID) : int(pID) for (tID, pID) in towerList}
    return trackDict, towerDict

def main():
    # databaseName = "/home/henry/lazy/29delphes_events.db"
    databaseName = "/home/henry/lazy/h1bBatch1.db"
    fields = ["ID", "PT", "Eta", "Phi", "E"]
    trackList = trackTowerCreators(databaseName, fields)
    # trackList = readSelected(databaseName, fields, "Tracks")
    clusterForest = ClusterForest(trackList, deltaR=.8)
    clusterForest.pltAssignMothers()
    # jetList = clusterForest.sort_clusters()
    # return jetList
    # test_ClusterTree()


if __name__ == '__main__':
    pass #main()
