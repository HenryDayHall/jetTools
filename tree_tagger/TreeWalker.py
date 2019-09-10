# TODO Needs updates for uproot!
# Jets are formed using applyFastJet.cc
# this takes in the jets, clusters them and writes to the text form of parentedTree
# read in the data
# make the jets into parented trees 
# there should be a Jet class that is an extention of ParentedTree from nltk
# use Scikit learn RobustScalar to rescale the data
import numpy as np
from matplotlib import pyplot as plt
from ipdb import set_trace as st
from skhep import math as hepmath
import os
import csv
import torch

class TreeWalker:
    def __init__(self, eventWise, jet_name, jet_number, pseudojet_idx):
        self.eventWise = eventWise
        self.jet_name = jet_name
        self.jet_number = jet_number
        self.pseudojet_idx = pseudojet_idx
        self.left_id = getattr(eventWise, jet_name+"_Child1")[jet_number][pseudojet_idx]
        self.right_id = getattr(eventWise, jet_name+"_Child2")[jet_number][pseudojet_idx]
        self.is_leaf = self.left_id < 0 and self.right_id < 0
        self.is_root = getattr(eventWise, jet_name+"_Parent")[jet_number][pseudojet_idx] < 0
        self.particle_idx = -1
        if self.is_leaf:
            self.particle_idx = eventWise.JetInputs_SelectedIdx[getattr(eventWise, jet_name+"_InputIdx")[jet_number][pseudojet_idx]]
        self.label = getattr(eventWise, jet_name + "_JoinDistance")[jet_number][pseudojet_idx]
        self.leaf = [jet._floats[node_index][i] for i in
                      [jet.pt_col, jet.rap_col, jet.phi_col, jet.energy_col]]
        self.pt = self.leaf[0]
        self.rap = self.leaf[1]
        self.phi = self.leaf[2]
        self.e = self.leaf[3]
        # this si the variabel offered to the nn
        # momentum rapidity theta phi energy transverse-momentum
        leaf_inputs = [jet.p, self.rap, jet.theta, jet.phi, self.e, self.pt]
        self.leaf_inputs = torch.DoubleTensor(leaf_inputs)
        # for visulisation
        self.size = self.e
        green = (0.2, 0.9, 0.15, 1.)
        blue = (0.1, 0.5, 0.85, 1.)
        self.colour = green if self.is_leaf else blue
        self._decendants = None
        # changing to holding the whole tree inn memory
        if not self.is_leaf:
            self.left = TreeWalker(self.jet, self.left_id)
            self.right = TreeWalker(self.jet, self.right_id)

#    @property
#    def left(self):
#        return TreeWalker(self.jet, self.left_id)
#
#    @property
#    def right(self):
#        return TreeWalker(self.jet, self.right_id)

    @property
    def decendants(self):
        if self._decendants is None:
            if self.is_leaf:
                self._decendants = set([self.global_obs_id])
            else:
                left_decendants = self.left.decendants
                right_decendants = self.right.decendants
                self._decendants = left_decendants.union(right_decendants)
        return self._decendants

# how abut a graph of average join properties
def join_behaviors(root):
    if root.is_leaf:
        return [], []
    else:
        # the phi coordinate is modular
        phi_step = root.left.phi - root.right.phi
        adjusted_phi_step = ((phi_step+np.pi)%(2*np.pi)) - np.pi
        modular_jump = abs(phi_step) > np.pi
        displacement = [root.left.rap - root.right.rap,
                        adjusted_phi_step,
                        root.left.pt - root.right.pt]
        displacement_left, jump_left  = join_behaviors(root.left)
        displacement_right, jump_right = join_behaviors(root.right)
        all_dsiplacement = displacement_left + displacement_right + [displacement]
        all_jump = jump_left + jump_right + [modular_jump]
        return all_dsiplacement, all_jump


# its impossible to to this properly retrospectivly... 
# the clustering has time order to it
def tree_motion(start, root, steps_between):
    if root.is_leaf:
        location = [[[start[0]], [start[1]]]]*(steps_between+1)
        size = [[root.size]]*(steps_between+1)
        colour = [[root.colour]]*(steps_between+1)
        return location, size, colour
    else:
        # find out what is beflow this point
        next_left = [root.left.rap, root.left.phi]
        below_left, below_left_size, below_left_colour = tree_motion(next_left, root.left, steps_between)
        next_right = [root.right.rap, root.right.phi]
        below_right, below_right_size, below_right_colour = tree_motion(next_right, root.right, steps_between)
        # if either were leaves pad to the right depth
        if len(below_left) < len(below_right):
            pad_height = int(len(below_right)-len(below_left))
            pad = pad_height * [below_left[-1]]
            below_left += pad
            pad_colour = pad_height * [below_left_colour[-1]]
            below_left_colour += pad_colour
            pad_size = pad_height * [below_left_size[-1]]
            below_left_size += pad_size
        elif len(below_right) < len(below_left):
            pad_height = int(len(below_left)-len(below_right))
            pad = pad_height * [below_right[-1]]
            below_right += pad
            pad_colour = pad_height * [below_right_colour[-1]]
            below_right_colour += pad_colour
            pad_size = pad_height * [below_right_size[-1]]
            below_right_size += pad_size
        assert len(below_left) == len(below_right)
        levels = [[r_rap+l_rap, r_phi+l_phi] for (r_rap, r_phi), (l_rap, l_phi)
                  in zip(below_left, below_right)]
        sizes = [l_size + r_size for l_size, r_size in zip(below_left_size, below_right_size)]
        colours = [l_col + r_col for l_col, r_col in zip(below_left_colour, below_right_colour)]
        # now make the this level
        rap_left = np.linspace(next_left[0], start[0], steps_between)
        rap_right = np.linspace(next_right[0], start[0], steps_between)
        # this coordinate is cyclic
        # so this next bit is a shuffle to get it to link by the shortest route
        distance = next_left[1] - start[1]
        if distance > np.pi:
            adjusted_next_left = -next_left[1]
        elif distance < -np.pi:
            adjusted_next_left = next_left[1] + 2*np.pi
        else:
            adjusted_next_left = next_left[1]
        phi_left = np.linspace(adjusted_next_left, start[1], steps_between)
        phi_left = ((phi_left+np.pi)%(2*np.pi)) - np.pi
        distance = next_right[1] - start[1]
        if distance > np.pi:
            adjusted_next_right = -next_right[1]
        elif distance < -np.pi:
            adjusted_next_right = next_right[1] + 2*np.pi
        else:
            adjusted_next_right = next_right[1]
        phi_right = np.linspace(adjusted_next_right, start[1], steps_between)
        phi_right = ((phi_right+np.pi)%(2*np.pi)) - np.pi
        this_level = [[[eleft, eright], [pleft, pright]]
                      for eleft, eright, pleft, pright in
                      zip(rap_left, rap_right, phi_left, phi_right)]
        this_level += [[[root.rap], [root.phi]]]
        # add all the motion together
        levels += this_level
        # pick size
        l_size = root.left.size; r_size = root.right.size
        this_size = [[l_size, r_size]]*steps_between + [[root.size]]
        sizes += this_size
        l_colour = root.left.colour; r_colour = root.right.colour
        this_colour = [[l_colour, r_colour]]*steps_between + [[(*root.colour[:3], 0.5)]]
        colours += this_colour
        return levels, sizes, colours


def plot_motions(motions, sizes, colours, dir_name, step_interval):
    os.mkdir(dir_name)
    # get the right x limits
    for motion in motions:
        plt.scatter(*motion[0])
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    # some motions may be shorter than others, extend the short ones
    max_len = max([len(m) for m in motions])
    for motion, size, colour in zip(motions, sizes, colours):
        deficit = max_len - len(motion)
        motion += [motion[-1]]*deficit
        size += [size[-1]]*deficit
        colour += [colour[-1]]*deficit
    # figure out how much padding we will need on the name
    end_extend = 10
    last_num = str(max_len+end_extend)
    pad_length = len(last_num)
    name_format = os.path.join(dir_name, f"frame{{:0{pad_length}d}}.png")
    # loop over the frames
    for level_n in range(max_len):
        plt.cla()
        for motion, size, colour in zip(motions, sizes, colours):
            level = motion[level_n]
            c = colour[level_n]
            s = size[level_n]
            plt.scatter(*level, c=c, s=np.sqrt(s), alpha=0.5)
        plt.xlabel("$rapidity$")
        plt.ylabel("$\\phi$")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(name_format.format(level_n))
    # save the last image a few times
    for j in range(level_n, level_n+end_extend):
        plt.savefig(name_format.format(j))

        
def quick_vid():
    import FormJets
    save_name = "homepic" 
    assert not os.path.exists(save_name)
    jet = FormJets.PsudoJets.read("test")
    walker = TreeWalker(jet, jet.root_psudojetIDs[0])
    steps_between = 10
    motion, size, colour = tree_motion(walker.leaf[1:3], walker, steps_between)
    plot_motions([motion], [size], [colour], save_name, steps_between)


def whole_event(nodisplay=False):
    import FormJets
    import Components
    obs_dir = "test"
    fast_save_name = "fasteventpic"
    home_save_name = "homeeventpic"
    assert not os.path.exists(fast_save_name)
    assert not os.path.exists(home_save_name)
    observables = Components.Observables.from_file(obs_dir)
    deltaR = 1.
    exponent_multiplyer = -1
    steps_between = 10
    print("Starting fastjet")
    fast_jets = FormJets.run_FastJet(obs_dir, deltaR, exponent_multiplyer)
    fast_jets = fast_jets.split()
    motions = []; colours = []; sizes = []
    for i, jet in enumerate(fast_jets):
        print(f"Getting fast motion. Jet {i}")
        fast_walker = TreeWalker(jet, jet.root_psudojetIDs[0])
        motion, size, color = tree_motion(fast_walker.leaf[1:3], fast_walker, steps_between)
        motions.append(motion); sizes.append(size); colours.append(color)
    if nodisplay:
        for i, motion in enumerate(motions):
            motions_file_name = f"fast_motions{i}.csv"
            with open(motions_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(motion)
        for i, colour in enumerate(colours):
            colours_file_name = f"fast_colours{i}.csv"
            with open(colours_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(colour)
        for i, size in enumerate(sizes):
            sizes_file_name = f"fast_sizes{i}.csv"
            with open(sizes_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(size)
    else:
        print("Plotting fast jets")
        plot_motions(motions, sizes, colours, fast_save_name, steps_between)
    home_jets = FormJets.PsudoJets(observables, deltaR, exponent_multiplyer)
    home_jets.assign_mothers()
    home_jets = home_jets.split()
    motions = []; colours = []; sizes = []
    for i, jet in enumerate(home_jets):
        print(f"Getting home motion. Jet {i}")
        home_walker = TreeWalker(jet, jet.root_psudojetIDs[0])
        motion, size, color = tree_motion(home_walker.leaf[1:3], home_walker, steps_between)
        motions.append(motion); sizes.append(size); colours.append(color)
    if nodisplay:
        for i, motion in enumerate(motions):
            motions_file_name = f"home_motions{i}.csv"
            with open(motions_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(motion)
        for i, colour in enumerate(colours):
            colours_file_name = f"home_colours{i}.csv"
            with open(colours_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(colour)
        for i, size in enumerate(sizes):
            sizes_file_name = f"home_sizes{i}.csv"
            with open(sizes_file_name, 'w') as mfile:
                writer = csv.writer(mfile)
                writer.writerows(size)
    else:
        print("Plotting home jets")
        plot_motions(motions, sizes, colours, home_save_name, steps_between)
    print("Done!")


def whole_event_behavior(nodisplay=False):
    import FormJets
    import Components
    obs_dir = "test"
    observables = Components.Observables.from_file(obs_dir)
    deltaR = 1.
    exponent_multiplyer = -1
    steps_between = 10
    print("Starting fastjet")
    fast_jets = FormJets.run_FastJet(obs_dir, deltaR, exponent_multiplyer)
    fast_jets = fast_jets.split()
    fast_behavior = []
    fast_jump = []
    for i, jet in enumerate(fast_jets):
        print(f"Getting fast behavior. Jet {i}")
        fast_walker = TreeWalker(jet, jet.root_psudojetIDs[0])
        jet_behavior, jet_jump = join_behaviors(fast_walker)
        fast_behavior += jet_behavior
        fast_jump += jet_jump
    fast_behavior = np.array(fast_behavior)
    fast_jump = np.array(fast_jump)
    home_jets = FormJets.PsudoJets(observables, deltaR, exponent_multiplyer)
    home_jets.assign_mothers()
    home_jets = home_jets.split()
    home_behavior = []
    home_jump = []
    for i, jet in enumerate(home_jets):
        print(f"Getting home behavior. Jet {i}")
        home_walker = TreeWalker(jet, jet.root_psudojetIDs[0])
        jet_behavior, jet_jump = join_behaviors(home_walker)
        home_behavior += jet_behavior
        home_jump += jet_jump
    home_behavior = np.array(home_behavior)
    home_jump = np.array(home_jump)
    print("Done!")
    if nodisplay:
        np.savetxt("fast_jump.csv", fast_jump)
        np.savetxt("fast_behavior.csv", fast_behavior)
        np.savetxt("home_jump.csv", home_jump)
        np.savetxt("home_behavior.csv", home_behavior)
    else:
        plot_whole_event_behavior(fast_behavior, fast_jump, home_behavior, home_jump, exponent_multiplyer, deltaR)

def plot_whole_event_behavior(fast_behavior, fast_jump, home_behavior, home_jump, exponent_multiplyer, deltaR):
    plt.scatter(fast_behavior[fast_jump, 0], fast_behavior[fast_jump, 1], c= fast_behavior[fast_jump, 2], cmap='viridis', marker='P', label=f"Fast jet, modular jump ({sum(fast_jump)} points)", edgecolor='k')
    plt.scatter(home_behavior[home_jump, 0], home_behavior[home_jump, 1], c= home_behavior[home_jump, 2], cmap='viridis', marker='o', label=f"Home jet, modular jump ({sum(home_jump)} points)", edgecolor='k')
    plt.scatter(fast_behavior[~fast_jump, 0], fast_behavior[~fast_jump, 1], c= fast_behavior[~fast_jump, 2], cmap='viridis', marker='P', label="Fast jet")
    plt.scatter(home_behavior[~home_jump, 0], home_behavior[~home_jump, 1], c= home_behavior[~home_jump, 2], cmap='viridis', marker='o', label="Home jet")
    plt.legend()
    # colourbar
    plt.colorbar(label="$p_T$ difference")
    # title
    if exponent_multiplyer == -1:
        algorithm = "anti-kt"
    elif exponent_multiplyer == 0:
        algorithm = "cambridge-aachen"
    else:
        algorithm = "kt"
    plt.title(f"Displacement between joined tracks. Algorithm = {algorithm}, R={deltaR}")
    # plot a circle at deltaR
    xs = np.cos(np.linspace(0, 2*np.pi, 50))*deltaR
    ys = np.sin(np.linspace(0, 2*np.pi, 50))*deltaR
    plt.plot(xs, ys, c='k')
    # axis
    plt.xlabel("rapidity")
    plt.ylabel("$\\phi$")
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    #quick_vid()
    #whole_event()
    whole_event_behavior()

