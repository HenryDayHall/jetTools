""" A script to check the contents/status of all data in a folder """
import os
from tree_tagger import Components, FormJets

def check_path(path):
    # check for fragments, these would contain raw jets
    fragment_dir = path[:-5] + "_fragment"
    if os.path.exists(fragment_dir):
        in_fragments = os.listdir(fragment_dir)
        jets_in_fragments = 0
        for name in in_fragments:
            frag = Components.EventWise.from_file(os.path.join(fragment_dir, name))
            jets_in_fragments += len(FormJets.get_jet_names(frag))
        del frag
    else:
        jets_in_fragments = -1
    # check it can be loaded
    try:
        ew = Components.EventWise.from_file(path)
        local_jets = len(FormJets.get_jet_names(ew))
        working = True
        git_message = ew.git_properties.latest_message
    except Exception:  # any error at all here means there is a break
        working = False
        local_jets = -1
        scored_jets = -1
    else:
        scored_jets = len([name for name in ew.hyperparameter_columns
                           if name.endswith("SeperateAveSignalMassRatio")])
    return working, local_jets, scored_jets, jets_in_fragments, git_message


def check_dir(dir_name="megaIgnore"):
    col_names = "file_name, working, local_jets, scored_jets, jets_in_fragments, git_message"
    output = col_names + os.linesep
    for name in os.listdir(dir_name):
        if not name.endswith(".awkd"):
            continue
        evaluation = check_path(os.path.join(dir_name, name))
        row = [name] + [str(x) for x in evaluation]

        output += ", ".join(row) + os.linesep
    return output

        
if __name__ == '__main__':
    output = check_dir()
    output += check_dir("megaIgnore/best_scans1")
    output += check_dir("megaIgnore/writeup1_scans")
    output += check_dir("megaIgnore/writeup2_scans")
    print(output)
    with open("evaluation.csv", 'w') as outfile:
        outfile.write(output)

