import numpy as np
from matplotlib import pyplot as plt
from ReadSQL import readSelected
from ipdb import set_trace as st

class ClusterTree:
    def __init__(self, trackList, deltaR=.4, exponent_multiplyer=-1):
        self.trackList = trackList
        self.deltaR = deltaR
        self.exponent = 2 * exponent_multiplyer
        # each cluster is a dictionary with referance to it's daughters and it's parameters
        # the tracks make the intial clusters
        self.clusters = [{'pt':t[1], 'eta':t[2], 'phi':t[3],
                          'trackID': t[0],  # will be None for all other clustered objects
                          'clusterID': ID,  # each cluster will have a unique id
                          'daughters': None, 'mother': 'unassigned'}
                         for ID, t in enumerate(trackList)]
        # keep track of which clusters don't yet have a mother
        self.currently_avalible = len(self.clusters)
        self._calculateDistances()
    
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
                    distance = min(self.clusters[row]['pt']**self.exponent, self.clusters[column]['pt']**self.exponent) *\
                               ((self.clusters[row]['eta'] - self.clusters[column]['eta'])**2 +
                               (self.clusters[row]['phi'] - self.clusters[column]['phi'])**2)
                self.distances[row, column] = distance

    
    def _recalculateOne(self, cluster_num):
        for row in range(self.currently_avalible):
            column = cluster_num
            if column > row:
                row, column = column, row  # keep the upper triangular form
            if column == row:
                distance = self.clusters[row]['pt']**self.exponent * self.deltaR
            else:
                distance = min(self.clusters[row]['pt']**self.exponent, self.clusters[column]['pt']**self.exponent) *\
                           ((self.clusters[row]['eta'] - self.clusters[column]['eta'])**2 +
                           (self.clusters[row]['phi'] - self.clusters[column]['phi'])**2)
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
        plt.axis([-4., 4., -4., 4.])
        etas = [c['eta'] for c in self.clusters]
        phis = [c['phi'] for c in self.clusters]
        plt.scatter(etas, phis, marker='x')
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self.distances), self.distances.shape)
            if row == column:
                decendents = self.getLastDecendants(self.clusters[row])
                etas = [d['eta'] for d in decendents]
                phis = [d['phi'] for d in decendents]
                plt.scatter(etas, phis)
                print(f"Added jet of {len(decendents)} tracks, {self.currently_avalible} clusters unfinished")
                plt.pause(0.05)
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

    def getLastDecendants(self, cluster=None, clusterID=None):
        if clusterID is None and cluster is None:
            raise TypeError("Need to specify a cluster")
        elif cluster is None:
            cluster = self.cluster_from_ID(clusterID)
        decendents = []
        # bu this point we have the first cluster
        if cluster['daughters'] is None:
            # just the one
            return [cluster]
        toCheck = [daughter for daughter in cluster['daughters']]
        while len(toCheck) > 0:
            cluster = self.cluster_from_ID(toCheck.pop())
            if cluster['daughters'] is None:
                decendents.append(cluster)
            else:
                toCheck += cluster['daughters']
        return decendents

            

    def _combine(self, cluster_index1, cluster_index2):
        new_index = len(self.clusters)
        self.clusters[cluster_index1]['mother'] = new_index
        self.clusters[cluster_index2]['mother'] = new_index
        cluster1 = self.clusters[cluster_index1]
        cluster2 = self.clusters[cluster_index2]
        new_cluster = {'trackID': None, 'daughters': [cluster1['clusterID'], cluster2['clusterID']],
            'mother': 'unassigned', 'clusterID': len(self.clusters)}
        cluster1 = self.clusters[cluster_index1]
        cluster2 = self.clusters[cluster_index2]
        new_cluster['pt'] = cluster1['pt'] + cluster2['pt']
        new_cluster['eta'] = (cluster1['pt']*cluster1['eta'] + cluster2['pt']*cluster2['eta'])/new_cluster['pt']
        new_cluster['phi'] = (cluster1['pt']*cluster1['phi'] + cluster2['pt']*cluster2['phi'])/new_cluster['pt']
        return new_cluster


def test_ClusterTree():
    # equlatral triangle
    eta = [0, 1, 2]
    phi = [0, np.sqrt(3), 0]
    ids = [0, 1, 2]
    pts = [1, 1, 1]
    trackList = np.array([ids, pts, eta, phi]).T
    clust = ClusterTree(trackList)
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
    


def main():
    databaseName = "/home/henry/lazy/29delphes_events.db"
    fields = ["ID", "PT", "Eta", "Phi"]
    trackList = readSelected(databaseName, fields, "Tracks")
    clusterTree = ClusterTree(trackList, deltaR=.8)
    clusterTree.pltAssignMothers()
    # test_ClusterTree()


if __name__ == '__main__':
    main()
    # clearly something here is broken.
