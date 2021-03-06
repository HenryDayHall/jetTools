from numpy import testing as tst
import numpy as np
from jet_tools import ReadHepmc, PDGNames
from test.tools import generic_equality_comp, TempTestDir, data_dir
#from ipdb import set_trace as st
import os
# overall momentum is conserved, but internal to the particle shower it isn't conserved

def void_test_ReadHepmc():
    hepmc_file = os.path.join(data_dir, "mini.hepmc")
    n_events = 1
    hepmc = ReadHepmc.Hepmc(hepmc_file, 0, n_events)
    idents = PDGNames.Identities()
    all_ids = set(idents.particle_data[:, idents.columns["id"]])
    for event_n in range(n_events):  # loop over out selection of events
        hepmc.selected_index = event_n
        # sanity checks on the components
        assert hepmc.Event_n == event_n
        # check the beam particles
        beam_idx1 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle1)[0][0]
        beam_idx2 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle2)[0][0]
        assert hepmc.Is_root[beam_idx1]
        assert hepmc.Is_root[beam_idx2]
        assert len(hepmc.Parents[beam_idx1]) == 0
        assert len(hepmc.Parents[beam_idx2]) == 0
        # valid PID
        assert set(np.abs(hepmc.MCPID)).issubset(all_ids)
        # conservation of momentum and parent child reflection
        num_particles = len(hepmc.MCPID)
        parent_p4 = np.zeros(4)
        child_p4 = np.zeros(4)
        self_p4 = np.zeros(4)
        children_issues = 0
        children_relations = 0
        parent_issues = 0
        parent_relations = 0
        conserved_parent_multiplicity = []
        conserved_children_multiplicity = []
        unconserved_parent_multiplicity = []
        unconserved_children_multiplicity = []
        for idx in range(num_particles):  # loop over particles in event
            parent_p4[:] = 0
            child_p4[:] = 0
            self_p4[:] = (hepmc.Energy[idx], hepmc.Px[idx], hepmc.Py[idx], hepmc.Pz[idx])
            if not hepmc.Is_root[idx]:
                parent_relations += 1
                # we have already checked the parents of roots
                parent_idx = hepmc.Parents[idx]
                assert len(parent_idx) > 0
                for pidx in parent_idx:
                    assert idx in hepmc.Children[pidx]
                    parent_p4 += (hepmc.Energy[pidx], hepmc.Px[pidx], hepmc.Py[pidx], hepmc.Pz[pidx])
                    # if these parents have children besides this child
                    # subtract their momentum out
                    for cidx in hepmc.Children[pidx]:
                        if cidx == idx:
                            continue
                        else:
                            parent_p4 -= (hepmc.Energy[cidx], hepmc.Px[cidx], hepmc.Py[cidx], hepmc.Pz[cidx])
                try:
                    tst.assert_allclose(parent_p4[1:], self_p4[1:], rtol=0.05, atol=0.001)
                    conserved_parent_multiplicity.append(len(parent_idx))
                except AssertionError as e:
                    #print(f"Event {event_n}, particle {idx}, parents; {e}")
                    parent_issues += 1
                    unconserved_parent_multiplicity.append(len(parent_idx))
            if hepmc.Is_leaf[idx]:
                assert len(hepmc.Children[idx]) == 0
            else:
                children_relations += 1
                children_idx = hepmc.Children[idx]
                assert len(children_idx) > 0
                for cidx in children_idx:
                    assert idx in hepmc.Parents[cidx]
                    child_p4 += (hepmc.Energy[cidx], hepmc.Px[cidx], hepmc.Py[cidx], hepmc.Pz[cidx])
                    # if these children have parents besides this one
                    # subtract their momentum out
                    for pidx in hepmc.Parents[cidx]:
                        if pidx == idx:
                            continue
                        else:
                            child_p4 -= (hepmc.Energy[pidx], hepmc.Px[pidx], hepmc.Py[pidx], hepmc.Pz[pidx])
                try:
                    tst.assert_allclose(child_p4[1:], self_p4[1:], rtol=0.05, atol = 0.001)
                    conserved_children_multiplicity.append(len(children_idx))
                except AssertionError as e:
                    #print(f"Event {event_n}, particle {idx}, children; {e}")
                    children_issues += 1
                    unconserved_children_multiplicity.append(len(children_idx))
        print(f"Event {event_n}")
        print(f"Children non-conservation {children_issues}; {100*children_issues/children_relations:.1f}%")
        print(f"Parent non-conservation {parent_issues}; {100*parent_issues/parent_relations:.1f}%")
        print(f"Any non-conservation {100*(parent_issues+children_issues)/(children_relations+parent_relations):.1f}%")
        print(f"Children multipicity, conserved {np.mean(conserved_children_multiplicity):.2f}, unconserved {np.mean(unconserved_children_multiplicity):.2f}")
        print(f"Parent multipicity, conserved {np.mean(conserved_parent_multiplicity):.2f}, unconserved {np.mean(unconserved_parent_multiplicity):.2f}")


def void_test_ReadHepmc2():
    hepmc_file = os.path.join(data_dir, "billy_tag_2_pythia8_events.hepmc")
    n_events = 4
    hepmc = ReadHepmc.Hepmc(hepmc_file, 0, n_events)
    idents = PDGNames.Identities()
    mom_diff = []
    all_ids = set(idents.particle_data[:, idents.columns["id"]])
    for event_n in range(n_events):  # loop over out selection of events
        hepmc.selected_index = event_n
        # sanity checks on the components
        assert hepmc.Event_n == event_n
        # check the beam particles
        beam_idx1 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle1)[0][0]
        beam_idx2 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle2)[0][0]
        assert hepmc.Is_root[beam_idx1]
        assert hepmc.Is_root[beam_idx2]
        assert len(hepmc.Parents[beam_idx1]) == 0
        assert len(hepmc.Parents[beam_idx2]) == 0
        # valid PID
        assert set(np.abs(hepmc.MCPID)).issubset(all_ids)
        # conservation of momentum at the vertices
        parent_p4 = np.zeros(4)
        child_p4 = np.zeros(4)
        issues = 0
        relations = 0
        conserved_multiplicity = []
        unconserved_multiplicity = []
        for v_barcode in hepmc.Vertex_barcode:
            parent_p4[:] = 0
            child_p4[:] = 0
            parents_idx = np.where(hepmc.End_vertex_barcode == v_barcode)[0]
            children_idx = np.where(hepmc.Start_vertex_barcode == v_barcode)[0]
            if len(parents_idx) == 0:
                for cidx in children_idx:
                    assert hepmc.Is_root[cidx]
            elif len(children_idx) == 0:
                for pidx in parents_idx:
                    assert hepmc.Is_leaf[pidx]
            else:
                for pidx in parents_idx:
                    parent_p4 += (hepmc.Energy[pidx], hepmc.Px[pidx], hepmc.Py[pidx], hepmc.Pz[pidx])
                for cidx in children_idx:
                    child_p4 += (hepmc.Energy[cidx], hepmc.Px[cidx], hepmc.Py[cidx], hepmc.Pz[cidx])
                relations += 1
                mom_diff.append(parent_p4 - child_p4)
                try:
                    tst.assert_allclose(parent_p4[1:], child_p4[1:], rtol=0.05, atol=0.001)
                    conserved_multiplicity.append(len(children_idx) + len(parents_idx))
                except AssertionError as e:
                    if issues < 1 and event_n==0:
                        parent_barcodes = hepmc.Particle_barcode[parents_idx]
                        children_barcodes = hepmc.Particle_barcode[children_idx]
                        print(f"Vertex {v_barcode} has incoming particles with barcodes {parent_barcodes} and outgoing particles with barcodes {children_barcodes}")
                        print(f"The incoming total momentum is {parent_p4}")
                        print(f"The outgoing total momentum is {child_p4}")
                    issues += 1
                    unconserved_multiplicity.append(len(children_idx) + len(parents_idx))
        print(f"Event {event_n}")
        print(f"non-conservation {issues}; {100*issues/relations:.1f}%")
        print(f"multipicity, conserved {np.mean(conserved_multiplicity):.2f}, unconserved {np.mean(unconserved_multiplicity):.2f}")
    return np.array(mom_diff)


def test_ReadHepmc():
    check_lines()
    hepmc_file = os.path.join(data_dir, "mini.hepmc")
    n_events = 4
    # try starting from 0
    hepmc1 = ReadHepmc.Hepmc(hepmc_file, 0, n_events)
    # try startng from 1
    start2 = 1
    hepmc2 = ReadHepmc.Hepmc(hepmc_file, start=start2, stop=n_events)
    idents = PDGNames.Identities()
    all_ids = set(idents.particle_data[:, idents.columns["id"]])
    for event_n in range(n_events):  # loop over out selection of events
        hepmc1.selected_index = event_n
        # sanity checks on the components
        assert hepmc1.Event_n == event_n
        # again for the other one
        check_event_consistancy(hepmc1, all_ids)
        try:
            hepmc2.selected_index = event_n
            assert hepmc2.Event_n == event_n + 1
        except IndexError:
            assert event_n + start2 == n_events
            continue  # reach the end of 1
        check_event_consistancy(hepmc2, all_ids)


def check_event_consistancy(hepmc, all_ids):
    # check the beam particles
    beam_idx1 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle1)[0][0]
    beam_idx2 = np.where(hepmc.Particle_barcode == hepmc.Barcode_beam_particle2)[0][0]
    assert hepmc.Is_root[beam_idx1]
    assert hepmc.Is_root[beam_idx2]
    assert len(hepmc.Parents[beam_idx1]) == 0
    assert len(hepmc.Parents[beam_idx2]) == 0
    # valid PID
    assert set(np.abs(hepmc.MCPID)).issubset(all_ids)
    # conservation of momentum and parent child reflection
    num_particles = len(hepmc.MCPID)
    root_p4 = np.zeros(4)
    leaf_p4 = np.zeros(4)
    for idx in range(num_particles):  # loop over particles in event
        if hepmc.Is_root[idx]:
            root_p4 += (hepmc.Energy[idx], hepmc.Px[idx], hepmc.Py[idx], hepmc.Pz[idx])
        if hepmc.Is_leaf[idx]:
            leaf_p4 += (hepmc.Energy[idx], hepmc.Px[idx], hepmc.Py[idx], hepmc.Pz[idx])
    tst.assert_allclose(root_p4, leaf_p4, atol=0.0001)


def check_lines():
    particles = []
    particles.append({'barcode': 3, 'Start_vertex_barcode': -1, 'End_vertex_barcode': -3, 'PID': 2, 'Px': 0., 'Py': 0., 'Pz': 7.3759823919779751e+02, 'Energy': 7.3759823919779751e+02})
    particles.append({'barcode': 9, 'Start_vertex_barcode': -1, 'End_vertex_barcode': -7, 'PID': 21, 'Px': -1.4386555718774256e+01, 'Py': -1.6927530140503261e+01, 'Pz': 4.6864615737194424e+02, 'Energy': 4.6917239377610036e+02})
    particles.append({'barcode': 4, 'Start_vertex_barcode': -2, 'Pz':58.104791987011104, 'End_vertex_barcode': -3, 'PID': -2, 'Px': 0, 'Py': 0, 'Pz': -5.1893763547975027e+00, 'Energy': 5.1893763547975027e+00})
    particles.append({'barcode': 5, 'Start_vertex_barcode': -3, 'End_vertex_barcode': -6, 'PID': 35, 'Px': 0, 'Py': 7.1054273576000000e-15, 'Pz': 7.3240886284999999e+02, 'Energy': 7.4278761555910899e+02})
    particles.append({'barcode': 6, 'Start_vertex_barcode': -4, 'End_vertex_barcode': -1, 'PID': 2, 'Px': -3.5527136788005009e-15, 'Py': -3.5527136788005009e-15, 'Pz': 1.2709487353166983e+03, 'Energy': 1.2709487353166983e+03})
    particles.append({'barcode': 7, 'Start_vertex_barcode': -5, 'End_vertex_barcode': -2, 'PID': -2, 'Px': -3.5527136788005009e-15, 'Py': -3.5527136788005009e-15, 'Pz': -5.1893763547976164e+00, 'Energy': 5.1893763547975027e+00})
    particles.append({'barcode': 8, 'Start_vertex_barcode': -6, 'End_vertex_barcode': -13, 'PID': 35, 'Px': 1.4386555718906120e+01, 'Py': 1.6927530140658423e+01, 'Pz': 7.9711320159753188e+02, 'Energy': 8.0696571790253006e+02})
    # for h1bBatch2.hepmc
    #particles.append({'barcode': 3, 'Start_vertex_barcode': -1, 'End_vertex_barcode': -3,
    #    'PID': 1, 'Px': 0., 'Py': 0., 'Pz':58.104791987011104, 'Energy':58.104791987011104})
    #particles.append({'barcode': 4, 'Start_vertex_barcode': -2, 'End_vertex_barcode': -3,
    #    'PID': -1, 'Px': 0., 'Py': 0., 'Pz':-59.434584436371104, 'Energy':59.434584436371104})
    #particles.append({'barcode': 9, 'Start_vertex_barcode': -2, 'End_vertex_barcode': -10,
    #    'PID': 1, 'Px': -105.03226729224349, 'Py': -58.130810879967129, 'Pz':-739.62417574317283, 'Energy':749.30300852851167})
    #particles.append({'barcode': 5, 'Start_vertex_barcode': -3, 'End_vertex_barcode': -6,
    #    'PID': 35, 'Px': 0., 'Py': 0., 'Pz':-1.3297924496000000, 'Energy':117.53937642477460})
    #particles.append({'barcode': 6, 'Start_vertex_barcode': -4, 'End_vertex_barcode': -1,
    #    'PID': 1, 'Px': -1.2789769243681803*10**(-13), 'Py': -5.6843418860808015*10**(-14), 'Pz':58.104791987011055, 'Energy':58.104791987011112})
    #particles.append({'barcode': 14, 'Start_vertex_barcode': -4, 'End_vertex_barcode': -15,
    #    'PID': -1, 'Px': -5.7820152482327529, 'Py': -94.171736449210982, 'Pz': 362.01370390194961, 'Energy': 374.10664035673017})
    #particles.append({'barcode': 7, 'Start_vertex_barcode': -5, 'End_vertex_barcode': -2,
    #    'PID': 21, 'Px': 8.7041485130612273*10**(-14), 'Py': 6.5725203057809267*10**(-14), 'Pz': -876.93569382111320, 'Energy': 876.93569382111275})
    #particles.append({'barcode': 8, 'Start_vertex_barcode': -6, 'End_vertex_barcode': -9,
    #    'PID': 35, 'Px': 105.03226729368578, 'Py': 58.130810880765388, 'Pz':-79.206726092116511, 'Energy':185.73747728206331})
    hepmc_file = os.path.join(data_dir, "mini.hepmc")
    n_events = 1
    hepmc = ReadHepmc.Hepmc(hepmc_file, 0, n_events)
    hepmc.selected_index = 0
    for particle in particles:
        index = np.where(hepmc.Particle_barcode == particle['barcode'])[0][0]
        assert particle['Start_vertex_barcode'] == hepmc.Start_vertex_barcode[index]
        assert particle['End_vertex_barcode'] == hepmc.End_vertex_barcode[index]
        assert particle['PID'] == hepmc.MCPID[index]
        tst.assert_allclose([particle['Px'], particle['Py'], particle['Pz'], particle['Energy']],
                            [hepmc.Px[index], hepmc.Py[index], hepmc.Pz[index], hepmc.Energy[index]])


