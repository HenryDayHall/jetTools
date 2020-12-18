# some generic, unintresting imports
import numpy as np
import os
from jet_tools.src import InputTools, FormJets, Components
from ipdb import set_trace # to allow interactiverty

def interactive():
    """A function to allow the user to play with the variables"""
    try:  # if the user presses ctrl-c we want to go back to the script
        set_trace()
        # now have a play with the objects created :)
        # press c-enter when you're done
    except:
        pass
    print("...")

print("~~ Example of reading a hepmc file ~~")
hepmc_path = "/data/h1bBatch2.hepmc"
if os.path.exists(hepmc_path):
    print(f"Found a file to read in at {hepmc_path}")
else:
    message = "Please give the location of a hepmc file (can tab complete): "
    hepmc_path = InputTools.get_file_name(message, "hepmc").strip()
    if not os.path.exists(hepmc_path):
        print("Not a vaild file path")
        exit
print("This hepmc file will be read into an awkward array")
input("press enter to continue\n...")
print(" > from jet_tools.src import ReadHepmc")
from jet_tools.src import ReadHepmc
print(" > required_size = 10")
required_size = 10
print(" > eventWise = ReadHepmc.Hepmc(*os.path.split(hepmc_path), start=0, stop=required_size)")
eventWise = ReadHepmc.Hepmc(hepmc_path, start=0, stop=required_size)
print("If the chosen stop point is beyond the end of the hepmc (or equal to np.inf) "
      "then the whole file is read. "
      "The awkward array is wrapped in a cutsom object called an EventWise. "
      "If the hepmc file is very large it may be necessary to read it in chunks. "
      "The chunks can be combined using eventWise.combine (see the doc string for combine).")
print(" > from jet_tools.src.Components import EventWise")
from jet_tools.src.Components import EventWise
print("This class is structured around the format of event by event particle data.")
print("(for direct access to the awkward array use EventWise._column_contents)")
input("press enter to continue\n...")

print("The eventWise has an attribute called columns that lists all the data inside it. ")
print(" > print(eventWise.columns)")
print(eventWise.columns)
print("Most of the columns are discribed in the file column_descriptions.txt.")
input("press enter to continue\n...")

print("All these items can be accessed as attributes; e.g. > eventWise.Px")
print("Give it a try, and press `c` then enter to continue when you are done.")
interactive()

print("As it is often desirable to work on one event at a time. "
      "This can be done by changing eventWise.selected_index from None to the desired index.")
print(" > eventWise.selected_index = 3")
eventWise.selected_index = 3
print(" > print(eventWise.Parents)")
print(eventWise.Parents)
print("Now only data from the third event is shown. "
      "To undo this, set selected_index to None ")
print(" > eventWise.selected_index = None")
eventWise.selected_index = None
print("Give it a try, and press `c` when you are done")
interactive()

print("Now a quick note on saving this format. "
      "Reading a hepmc file takes time, "
      "so it is convenient to save this format directly. "
      "By default it will save as h1bBatch2_hepmc.awkd.")
new_path = hepmc_path[:-6] + "_hepmc.awkd"
print("this can be altered by changing the variable eventWise.save_name. ")
print("To save the object, use the write method;")
print(" > eventWise.write()")
eventWise.write()
print("This can be quickly read again by constructing an EventWise object. "
      "Using the class method EventWise.from_file;")
print(" > from_disk = EventWise.from_file(new_path)")
from_disk = EventWise.from_file(new_path)
input("press enter to continue\n...")

print("There are a number of other useful methods avalible from the EventWise class. "
      "fragment, split_unfinished, combine and recursive_combine all exist to "
      "break up large file for processing on multiple cores. "
      "append, remove, rename all exist to add and manipulate content. "
      "append_hyperparameters allows adding content that is common to all events in the set. "
      "match_indices is useful for doing more complex selections. "
      "See the doc strings for a full explanation. ")
input("press enter to continue\n...")

if InputTools.yesNo_question("Do you want to try adding a root file for detector content? "):
    print("~~ Example of reading a root file ~~")
    print(".root files produced by delphes can also be read into eventWise objects, "
          "other root files could too in principle, "
          "however, the code included in jetTools makes assumptions about the "
          "structure of the root file that corrispond to a root file written by delphes. "
          "This code also fixes various idiosyncrasies of delphes root files.")
    root_path = "/data/h1bBatch2.root"
    if os.path.exists(root_path):
        print(f"Found a file to read in at {root_path}")
    else:
        message = "Please give the location of a root file (can tab complete): "
        root_path = InputTools.get_file_name(message, "root").strip()
        if not os.path.exists(root_path):
            print("Not a vaild file path")
            exit
    input("press enter to continue\n...")

    print("The easiest way to use the RootReadout class is actually with "
          "the function marry in JoinHepMCRoot.py")
    print(" > from jet_tools.src import JoinHepMCRoot")
    from jet_tools.src import JoinHepMCRoot
    print(" > eventWise = JoinHepMCRoot.marry(hepmc_path, root_path)")
    eventWise = JoinHepMCRoot.marry(eventWise, root_path)
    print("This results in an eventWise that contains data from the root file "
          "and the hepmc file")
    root_present = True
else:
    root_present = False


print("It is often useful to calculate some values from the momentum and energy.")
print("In the Components module there are a number of methods for this; ")
print("  > Components.add_thetas(eventwise)")
print("  > Components.add_pt(eventwise)")
print("  > Components.add_phi(eventwise)")
print("  > Components.add_rapidity(eventwise)")
print("  > Components.add_pseudorapidity(eventwise)")
print("  > Components.add_mass(eventwise)")
print("These have been combined in Components.add_all(eventWise)")
print("  > Components.add_all(eventWise, inc_mass=True)")
Components.add_all(eventWise, inc_mass=True)
print("All these variables now exist as attributes of the eventWise, have a look; ")
interactive()

if InputTools.yesNo_question("Do you want to cluster the jets with Anti-kt? "):
    print("~~ Example of adding JetInputs ~~")
    if root_present:
        print("As the root file has been added we can form jets based on the particles")
        print("based on the particles that actually have been picked up by the detector.")
        print(" > filter1 = FormJets.filter_obs")
        filter1 = FormJets.filter_obs
    else:
        print("As there is no detector data to work on we will consider all particles that")
        print("exist at the end of the shower.")
        print(" > filter1 = FormJets.filter_ends")
        filter1 = FormJets.filter_ends
    input("press enter to continue\n...")
    print("Next, we select a filter that will remove particles with rapidity > 2.5")
    print("(high rapidities will miss the silicon tracker)")
    print("and PT < 0.5 (low PT is not reconstructed well enough to use)")
    filter2 = FormJets.filter_pt_eta
    input("press enter to continue\n...")
    print("After applying these filters isolate the particles that can be used to form jets;")
    print(" > FormJets.create_jetInputs(eventWise, [filter1, filter2], batch_length=np.inf)")
    FormJets.create_jetInputs(eventWise, [filter1, filter2], batch_length=np.inf)
    print("the eventwise now has various attributes that start with the word 'JetInputs_'")
    print("the clustering algorithm will look for these.")
    print(" > eventWise.JetInputs_Px")
    print(eventWise.JetInputs_Px)
    input("press enter to continue\n...")


    print("~~ Example of clustering a single event ~~")
    print("Anti-KT, Cambridge-Aachen and KT jet clustering can all be created with")
    print("FormJet.Traditonal which is a generalise method.")
    print("To cluster a single event with Anti-KT first select the event ")
    print(" > eventWise.selected_index = 0")
    eventWise.selected_index = 0
    print("Then create an object of type FormJet.Traditonal")
    print("traditonal_jets = FormJets.Traditional(eventWise, DeltaR=0.8, ExpofPTMultiplier=-1., assign=True)")
    traditonal_jets = FormJets.Traditional(eventWise, DeltaR=0.8, ExpofPTMultiplier=-1., assign=True)
    input("press enter to continue\n...")
    print("This object has clustered the event with anti-KT,")
    print("the key to this being anti-KT is the parameter ExpofPTMultiplier being -1.")
    print("ExpofPTMultiplier equal to 0 is Cambridge-Aachen and 1 is KT.")
    print("This object has some useful attributes;")
    print("This form of jet clustering puts the particles into binary trees")
    print("The form of these trees is given by the attributes")
    print("Parent, Child1, Child2")
    print(" > traditonal_jets.Parent")
    print(traditonal_jets.Parent)
    print("The numbers are local indices, so")
    print("the parent of the particle at index i is at index traditonal_jets.Parent[i]")
    input("press enter to continue\n...")
    print(" > traditonal_jets.PT")
    print("Then there are kinematic properties;")
    print(traditonal_jets.PT)
    print("This is the PT of all the input particles, and all the pseudojets")
    print("that where formed durng the clustering, and also the final jets.")
    print("Ps, Py, Pz, Rapidity, Phi, Energy can all be access in the same way")
    print("This is the PT of all the input particles, and all the pseudojets")
    print("that where formed durng the clustering, and also the final jets.")
    input("press enter to continue\n...")
    print("Normally you want to handle the jets indervidually,")
    print("this object, traditonal_jets, contains all the jets in the event.")
    print("To return a list of object that each contain one jet;")
    print(" > list_of_jets = traditonal_jets.split()")
    list_of_jets = traditonal_jets.split()
    interactive()

    print("~~ Example of using the jet clustering feature ~~")
    print("Normally you want to cluster all the events in your dataset.")
    print("It is also convenient to save the results back into the eventWise.")
    print("For this, the function FormJets.cluster_multiapply exists.")
    print("To use it, specify the clustering hyperparameters as a dict, so for anti-KT;")
    print(" > hyperparameters = {'DeltaR': 0.8, 'ExpofPTMultiplier': -1}")
    hyperparameters = {'DeltaR': 0.8, 'ExpofPTMultiplier': -1}
    print("Then we can run this clustering")
    print("FormJets.cluster_multiapply(eventWise, cluster_algorithm=FormJets.Traditional,")
    print("                            dict_jet_params=hyperparameters, jet_name='MyJet',")
    print("                            batch_length=np.inf)")
    FormJets.cluster_multiapply(eventWise, cluster_algorithm=FormJets.Traditional,
                                dict_jet_params=hyperparameters, jet_name="MyJet",
                                batch_length=np.inf)
    print("Now every event has been clustered with anti-KT.")
    print("The results are saved in the eventWise, and can be accessed as properties that")
    print("are named starting with <jet_name>_, so for us MyJet_")
    print(" > eventWise.selected_index = None")
    eventWise.selected_index = None
    print(" > eventWise.MyJet_Parent")
    print(eventWise.MyJet_Parent)
    print("Each event is a list of lists,")
    print("each sublist within an event is the information for 1 jet.")
    print("Something like; dataset{ event{ jet{} jet{}} event{ jet{}}}")
    interactive()
    print("Done")

