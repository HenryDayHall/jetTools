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
    from tree_tagger import ReadSQL
    event, track_list, tower_list, _ = ReadSQL.main()
    reco_collection = hit_detector(event.particle_list, tower_list, track_list)
    file_name = InputTools.getfilename("Save name (empty to cancel):")
    if file_name:
        reco_collection.write(file_name)

if __name__ == '__main__':
    main()

