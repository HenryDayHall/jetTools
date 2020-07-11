# some generic, unintresting imports
import os
from tree_tagger import InputTools
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
print(" > from tree_tagger import ReadHepmc")
from tree_tagger import ReadHepmc
print(" > required_size = 10")
required_size = 10
print(" > eventWise = ReadHepmc.Hepmc(*os.path.split(hepmc_path), start=0, stop=required_size)")
eventWise = ReadHepmc.Hepmc(*os.path.split(hepmc_path), start=0, stop=required_size)
print("If the chosen stop point is beyond the end of the hepmc (or equal to np.inf) "
      "then the whole file is read. "
      "The awkward array is wrapped in a cutsom object called an EventWise. "
      "If the hepmc file is very large it may be necessary to read it in chunks. "
      "The chunks can be combined using eventWise.combine (see the doc string for combine).")
print(" > from tree_tagger.Components import EventWise")
from tree_tagger.Components import EventWise
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
print(" > from tree_tagger import JoinHepMCRoot")
from tree_tagger import JoinHepMCRoot
print(" > eventWise = JoinHepMCRoot.marry(hepmc_path, root_path)")
# this takes lots of time, so don't actually do it
#eventWise = JoinHepMCRoot.marry(hepmc_path, root_path)
print("This results in an eventWise that contains data from the root file "
      "and the hepmc file")


