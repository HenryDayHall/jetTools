""" module to line up datasets by their hard events """
from ipdb import set_trace as st
from tree_tagger import InputTools, Components
import numpy as np
import awkward

def get_hard_kinematics(eventWise):
    heavy_pid = 35
    light_pid = 25
    eventWise.selected_index = None
    n_events = len(eventWise.X)
    all_kinematics = []
    for event_n in range(n_events):
        eventWise.selected_index = event_n
        # find the higgs particles
        heavy_idxs = set(np.where(eventWise.MCPID == heavy_pid)[0])
        light_idxs = set(np.where(eventWise.MCPID == light_pid)[0])
        # filter to only include one
        heavy_idx = next(i for i in heavy_idxs if not heavy_idxs.intersection(eventWise.Parents[i]))
        light_idxs = [i for i in light_idxs if not light_idxs.intersection(eventWise.Children[i])]
        hard_idxs = light_idxs + [heavy_idx]
        # add parents and children
        hard_idxs += eventWise.Parents[heavy_idx].tolist()
        for light_idx in light_idxs:
            hard_idxs += eventWise.Children[light_idx].tolist()
        # make the kinematics
        kinematics = np.vstack((eventWise.Energy[hard_idxs],
                                eventWise.Px[hard_idxs],
                                eventWise.Py[hard_idxs],
                                eventWise.Pz[hard_idxs]))
        all_kinematics.append(kinematics)
    all_kinematics = awkward.fromiter(all_kinematics)
    return all_kinematics


def next_jump(short_kinematics, long_kinematics, pre_jump_distances):
    n_values = len(short_kinematics)
    post_jump_distances = find_distances(short_kinematics, long_kinematics[1:])
    jump_at = 0
    while pre_jump_distances[jump_at] < post_jump_distances[jump_at]:
        jump_at += 1
        if jump_at >= len(short_kinematics):
            break
    return post_jump_distances[jump_at:], jump_at


def find_distances(short_kinematics, long_kinematics):
    distances = []
    for s_k, l_k in zip(short_kinematics, long_kinematics):
        if len(s_k[0]) == len(l_k[0]):
            distance = np.sum(np.abs((s_k - l_k).tolist()))
        else:
            distance = np.inf
        distances.append(distance)
    return distances


def find_jumps(short_kinematics, long_kinematics):
    n_jumps = len(long_kinematics) - len(short_kinematics)
    jumps = []
    # start by calculating every distance with no jumps
    pre_jump_distances = find_distances(short_kinematics, long_kinematics)
    # add the jumps one at a time
    long_start = 0
    short_start = 0
    while len(jumps) < n_jumps:
        pre_jump_distances, jump = next_jump(short_kinematics[short_start:],
                                             long_kinematics[long_start:],
                                             pre_jump_distances)
        long_start += jump
        jumps.append(jump)
    return jumps
        

def sequence_with_jumps(length, jumps):
    sequence = np.arange(length-len(jumps), dtype=int)
    for j in jumps:
        sequence[j:] += 1
    assert sequence[-1] + 1 == length
    return sequence


def apply_jumps(eventWise_full, eventWise_broken):
    eventWise_full.selected_index = eventWise_broken.selected_index = None
    short_kinematics = get_hard_kinematics(eventWise_broken)
    long_kinematics = get_hard_kinematics(eventWise_full)
    jumps = find_jumps(short_kinematics, long_kinematics)
    print(jumps)
    total_events = len(long_kinematics)
    event_numbers = sequence_with_jumps(total_events, jumps)
    eventWise_full.selected_index = eventWise_broken.selected_index = None
    eventWise_broken.append(Event_n=awkward.fromiter(event_numbers))

if __name__ == '__main__':
    path_full = InputTools.get_file_name("Complete eventWise file; ", '.awkd').strip()
    full = Components.EventWise.from_file(path_full)
    path_broken = InputTools.get_file_name("Incomplete eventWise file; ", '.awkd').strip()
    broken = Components.EventWise.from_file(path_broken)
    apply_jumps(full, broken)

