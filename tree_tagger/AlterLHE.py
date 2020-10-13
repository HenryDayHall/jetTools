"""To add particles to events in LHE files 


    Event format
 9      1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01
 "NUP", "IDPRUP", "XWGTUP", "SCALUP", "AQEDUP", "AQCDUP"
 "number of particles", ??, "event weight", "event scale", "alpha qed", "alpha qcd"
       -1 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +2.9011626398e+02 2.9011626398e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
 "IDUP", "ISTUP", "MOTHUP(1)", "MOTHUP(2)", "ICOLUP(1)", "ICOLUP(2)", "PUP(1)", "PUP(2)", "PUP(3)", "PUP(4)", "PUP(5)", "VTIMUP", "SPINUP"
 "MCPID", "status", "mother 1", "mother 2", "colour 1", "colour 2", "px", "py", "pz", "e", "m", "propper lifetime", "spin"
"""

from ipdb import set_trace as st
import os
from xml.etree import ElementTree
import ast
from tree_tagger import PDGNames, InputTools
import numpy as np

pdg_ids = PDGNames.Identities()


class Particle(dict):
    particle_attributes = "MCPID", "status", "mother 1", "mother 2", "colour 1", "colour 2", "px", "py", "pz", "e", "m", "propper lifetime", "spin"
    def __init__(self, particle_text=None):
        if particle_text is None:
            attributes = {attr: -1 for attr in self.particle_attributes}
        else:
            attributes = {attr: ast.literal_eval(value)
                          for attr, value in zip(self.particle_attributes,
                                                 particle_text.split())}
        self.update(attributes)

    def __str__(self):
        line = "  "*6 + " ".join([str(self.get(name)) for name in self.particle_attributes])
        return line
    

class Event:
    emissions = [21, 22]
    probabilities = np.array([2199, 15])/2214
    
    def __init__(self, event_text):
        lines = event_text.strip().split(os.linesep)
        self.event_info = " " + lines[0]
        self.particles = [Particle(line) for line in lines[1:]]

    def increment_particle_count(self, increase_by=1):
        split_info = self.event_info.split(maxsplit=1)
        current = int(split_info[0])
        self.event_info = f" {current+increase_by} {split_info[1]}"

    def __str__(self):
        return os.linesep.join([self.event_info] + [str(p) for p in self.particles])

    def add_split(self, split_type, max_split):
        b_idxs = [i for i, p in enumerate(self.particles) if abs(p["MCPID"]) == 5]
        b_idx = np.random.choice(b_idxs)
        b_particle = self.particles[b_idx]
        b_after = Particle(str(b_particle))  # copy it
        # the original b will be decayed
        b_particle["status"] = 2  # decayed
        # the new b_came from the first b
        b_after["mother 1"] = b_after["mother 2"] = b_idx + 1  # not 0 indexed
        # make a new particle
        new_particle = Particle()
        new_particle["MCPID"] = np.random.choice(self.emissions, p=self.probabilities)
        split_attributes = pdg_ids[new_particle["MCPID"]]
        new_particle["status"] = 1  # end state
        new_particle["mother 1"] = new_particle["mother 2"] = b_idx + 1
        # there are three undecided quantium numbers spin, colour and charge
        assert split_attributes["charge"] == 0, "need to implement non-zero charges"
        new_particle["charge"] = split_attributes["charge"]
        new_particle["m"] = 0.  # both the gluon and the photon are massless
        new_particle["propper lifetime"] = 0. # always seems to be 0 in lhe
        b_is_anti = b_particle["MCPID"] < 0
        if split_attributes["colType"] == 0:
            new_particle["colour 1"] = new_particle["colour 2"] = 0
        else:
            all_existing_colour = np.array([[p["colour 1"], p["colour 2"]] for p in self.particles],
                                           dtype=int)
            new_colour = np.max(all_existing_colour) + 1
            # one colour is taken from b
            if b_is_anti:
                new_particle["colour 2"] = b_particle["colour 2"]
                new_particle["colour 1"] = new_colour
                b_after["colour 2"] = new_colour
            else:
                new_particle["colour 1"] = b_particle["colour 1"]
                new_particle["colour 2"] = new_colour
                b_after["colour 1"] = new_colour
        # assume the spin will flip
        assert split_attributes['spin'] == 1.0, "need to implement splins besides 1"
        b_after["spin"] *= -1
        if b_particle["spin"] > 0:
            new_particle["spin"] = 1
        else:
            new_particle["spin"] = -1
        # now the kinematics must be taken
        parent_momentum = np.array([b_particle["px"], b_particle["py"], b_particle["pz"]])
        if split_type == "collinear":
            new_b_kinematics, new_other_kinematics = collinear_kinematics(parent_momentum, max_split)
        elif split_type == "soft":
            new_b_kinematics, new_other_kinematics = soft_kinematics(parent_momentum, max_split)
        # put in the kinematics of the other particle
        new_particle["px"], new_particle["py"], new_particle["pz"] = new_other_kinematics
        new_particle["e"] = np.sqrt(new_particle["m"]**2 + np.sum(new_other_kinematics**2))
        # the kinematics of the new particle
        b_after["px"], b_after["py"], b_after["pz"] = new_b_kinematics
        #b_after["e"] = np.sqrt(b_after["m"]**2 + np.sum(new_b_kinematics**2))
        b_after["e"] = b_particle["e"] - new_particle["e"]  # b can eb off shell
        # finnaly we can add the particles to the particle list
        self.particles += [b_after, new_particle]
        self.increment_particle_count(2)


def collinear_kinematics(parent_momentum, max_split):
    new_xyz = 0.5*np.tile(parent_momentum, (2, 1))
    change = max_split*(np.random.rand(3)-0.5)
    new_xyz[0] += change
    new_xyz[1] -= change
    return new_xyz


def soft_kinematics(parent_momentum, max_radiation):
    radiation = max_radiation*(np.random.rand(3)-0.5)
    return parent_momentum - radiation, radiation


def apply_to_events(old_name, new_name, split_type, end_split=5.):
    new_text = ""
    start_point = "<event>"
    start_length = len(start_point)
    i = 0
    with open(old_name, 'r') as old_file:
        old_text = old_file.read()
    num_events = old_text.count(start_point)
    inv_n_events = 1/num_events
    for i in range(num_events):
        next_event = old_text.index(start_point) + start_length
        new_text += old_text[:next_event]
        old_text = old_text[next_event:]
        event_end = old_text.index('<')
        # create the event
        event = Event(old_text[:event_end])
        event.add_split(split_type, i*inv_n_events*end_split)
        new_text += os.linesep + str(event) + os.linesep
        # cut the event off the old text
        old_text = old_text[event_end:]
    # add any remaining text on
    new_text += old_text
    with open(new_name, 'w') as new_file:
        new_file.write(new_text)



#def apply_to_events(old_name, new_name, split_type, end_split=5.):
#    tree = ElementTree.parse(old_name)
#    event_nodes = tree.findall('event')
#    inv_n_events = 1/len(event_nodes)
#    for i, event_node in enumerate(event_nodes):
#        event = Event(event_node.text)
#        event.add_split(split_type, i*inv_n_events*end_split)
#        event_node.text = str(event)
#    tree.write(new_name)
#

if __name__ == '__main__':
    old_name = InputTools.get_file_name("Name of lhe file to read ", '.lhe').strip()
    new_name = InputTools.get_file_name("Where to write the file? ", '.lhe').strip()
    split_type = InputTools.list_complete("What type of split? ", ['soft', 'collinear']).strip()
    apply_to_events(old_name, new_name, split_type)
    print("Done")

