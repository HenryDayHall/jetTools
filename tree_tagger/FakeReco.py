""" Skip the reco process and just pull all MC particles that hit the detector in any sense """
from tree_tagger import Components, PDGNames, InputTools
from ipdb import set_trace as st

def hit_detector(particle_list, tower_list, track_list):
    use_globals = [t.global_id for t in track_list]
    for tower in tower_list:
        use_globals += list(tower.global_ids)
    use_particles = [p for p in particle_list if p.global_id in use_globals]
    charge_map = PDGNames.Identities().charges
    reco_particles = [Components.RecoParticle.from_MC(p, i, charge_map)
                      for i, p in enumerate(use_particles)]
    reco_collection = Components.RecoParticleCollection(*reco_particles)
    return reco_collection

def main():
    from tree_tagger import ReadSQL, ReadHepmc
    hepmc_name = "/home/henry/lazy/h1bBatch2.hepmc"
    database_name = "/home/henry/lazy/h1bBatch2.db"
    events = ReadHepmc.read_file(hepmc_name, chatty=True)
    reco_collections = []
    mc_collections = []
    for i, event in enumerate(events):
        if i % 100 == 0:
            print(i, end=' ', flush=True)
        try:
            track_list, tower_list = ReadSQL.read_tracks_towers(event, database_name, i)
            reco_collection = hit_detector(event.particle_list, tower_list, track_list)
            mc_collection = Components.MCParticleCollection(*event.particle_list)
            reco_collections.append(reco_collection)
            mc_collections.append(mc_collection)
        except Exception as e:
            print(f"Problem with event {i}; {e}")
    reco_multi_collection = Components.MultiParticleCollections(reco_collections)
    reco_file_name = InputTools.getfilename("Save name for Reco (empty to cancel):")
    try:
        if reco_file_name:
            reco_multi_collection.write(reco_file_name)
    except Exception as e:
        print(e)
        st()
    mc_multi_collection = Components.MultiParticleCollections(mc_collections)
    mc_file_name = InputTools.getfilename("Save name for MC (empty to cancel):")
    try:
        if mc_file_name:
            mc_multi_collection.write(mc_file_name)
    except Exception as e:
        print(e)
        st()
    st()

if __name__ == '__main__':
    main()

