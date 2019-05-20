# needs sorting out, lots of imports missing, not updated for new components module

def showerClusterGrid(showers, jetClusters):
    # will need re writing when using real towers/tracks
    showerIDss = [np.hstack((shower.IDs[shower.makesTrack>0], shower.IDs[shower.makesTower>0]))
                  for shower in showers]
    n_jets = len(jetClusters)
    n_showers = len(showers)
    jetHits = np.zeros((n_jets, n_showers))
    for row, jet in enumerate(jetClusters):
        jetHits[row] = [sum(ID in jet.obsParticleIDs for ID in sIDs)
                          for sIDs in showerIDss]
    # now sort it onto the diagonal
    diagonal_jetHits = np.zeros((min(n_jets, n_showers), n_showers))
    shower_order = []
    # work through the rows in order of the biggest cluster first
    sort_order = np.array([-sum(hits) for hits in jetHits]).argsort()
    # if there are more jets than showers then cut to the number of showers avalible
    sort_order = sort_order[:n_showers]
    matched_jets = [None for _ in sort_order]
    for row in sort_order:
        shower_jetHits = jetHits[row]  # this will be the hits of each shower this jet got
        matching_order = list(shower_jetHits.argsort())  # indices from smallest to largest
        # cannot select the same shower twice
        while matching_order[-1] in shower_order:
            matching_order.pop()
        diagonal_jetHits[matching_order[-1]] = shower_jetHits
        matched_jets[matching_order[-1]] = jetClusters[row]
    return diagonal_jetHits, matched_jets


def make_showerClusterGrid(databaseName="/home/henry/lazy/29delphes_events.db", hepmc_name="/home/henry/lazy/29pythia8_events.hepmc"):
    # get the showers
    hepmc = Hepmc_event()
    hepmc.read_file(hepmc_name)
    hepmc.assign_heritage()
    showers = DrawTrees.getShowers(hepmc)
    for shower in showers:
        DrawTrees.addTracksTowers(databaseName, shower)
    # get the jets
    fields = ["ID", "PT", "Eta", "Phi", "E"]
    trackList = trackTowerCreators(databaseName, fields)
    clusterForest = ClusterForest(trackList, deltaR=.8)
    clusterForest.assignMothers()
    jetClusters = clusterForest.sort_clusters()
    # trackDict, towerDict = trackTowerDict(databaseName)
    for jet in jetClusters:
        jet.addObsLevel()
    diagonal_jetHits, matched_jets = showerClusterGrid(showers, jetClusters)
    return diagonal_jetHits, showers, matched_jets 

def plot_showerClusterGrid(databaseName="/home/henry/lazy/29delphes_events.db", hepmc_name="/home/henry/lazy/29pythia8_events.hepmc"):
    diagonal_jetHits, selected_showers, matched_jets = make_showerClusterGrid(databaseName, hepmc_name)
    # num_jets = 5
    # diagonal_jetHits = diagonal_jetHits[np.argsort(np.sum(diagonal_jetHits, axis=1))[-num_jets:]]
    shower_ticks = np.arange(len(selected_showers))
    shower_labels = [shower.flavour for shower in selected_showers]
    max_val = np.max(diagonal_jetHits) * 10
    min_val = 0.1
    plt.imshow(diagonal_jetHits, cmap='terrain_r', interpolation='nearest') #, norm=LogNorm(vmin=min_val, vmax=max_val))
    plt.xticks(shower_ticks, shower_labels, rotation=90)
    #plt.yticks(np.arange(num_jets), np.arange(num_jets))
    cbar = plt.colorbar()
    #cbar = plt.colorbar(orientation='horizontal', pad=0.4)
    cbar.set_label("Number of hits from shower in jet")
    plt.xlabel("Showers")
    plt.ylabel("Jets")
    plt.title("Jet shower alignment")
    plt.show()

def make_showerOverlap(databaseName="/home/henry/lazy/29delphes_events.db", hepmc_name="/home/henry/lazy/29pythia8_events.hepmc", percent=False):
    # get the showers
    hepmc = ReadHepmc.Hepmc_event()
    hepmc.read_file(hepmc_name)
    hepmc.assign_heritage()
    showers = get_showers(hepmc)
    for shower in showers:
        addTracksTowers(databaseName, shower)
    overlaps = np.zeros((len(showers), len(showers)))
    observable_masks = [np.logical_or(shower.makes_track, shower.makes_tower) for shower in showers]
    observable_global_ids = [set(shower.global_ids[mask])  for shower, mask in zip(showers, observable_masks)]
    for row, global_ids in enumerate(observable_global_ids):
        remaining = len(showers) - row -1
        overlaps[row] = [len(global_ids.intersection(otherglobal_ids)) for otherglobal_ids in observable_global_ids[:row + 1]]\
                        + [0 for _ in range(remaining)]
    if percent:
        for row, global_ids in enumerate(observable_global_ids):
            overlaps[row] /= len(global_ids)
    return showers, overlaps


def plot_showerOverlaps():
    showers, overlaps = make_showerOverlap(percent=True)
    shower_ticks = np.arange(len(showers))
    shower_labels = [shower.flavour for shower in showers]
    plt.imshow(overlaps, cmap='hot', interpolation='nearest')
    plt.xticks(shower_ticks, shower_labels, rotation=90)
    plt.yticks(shower_ticks, shower_labels)
    cbar = plt.colorbar()
    cbar.set_label("Percent shared hits in each shower")
    plt.xlabel("Showers")
    plt.ylabel("Showers")
    plt.title("Shower overlap")
    plt.show()

