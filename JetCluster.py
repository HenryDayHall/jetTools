
class ClusterTree:
    def __init__(self, trackList, deltaR=.4):
        self.trackList = trackList
        self.deltaR = deltaR
        # each cluster is a dictionary with referance to it's daughters and it's parameters
        # the tracks make the intial clusters
        self.clusters = [{'pt':t[1], 'theta':t[2], 'phi':t[3],
                          'trackID': t[0],  # will be None for all other clustered objects
                          'daughters': None, 'mothers': 'unassigned'}
                         for t in trackList]
        self.currently_avalible = [i for i, cluster in enumerate(clusters)]
    
    def _calculateDistances(self):
        pass

    def assignMothers(self):
        pass





