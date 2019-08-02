""" Module to run the linking NN """
import os
from tree_tagger import LinkingNN, LinkingEvaluation, RunTools, InputTools


def main():
    tst_dir = "./tst"
    in_tst = os.listdir(tst_dir)
    runs_in_tst = sorted([name for name in in_tst
                          if name.startswith("run") and name.endswith(".txt")],
                          key=lambda name: int(name[3:-4]))
    default = os.path.join(tst_dir, runs_in_tst[-1])
    print("Pick a run file;")
    user_choice = InputTools.getfilename(f"(Enter for {default}) ", file_ending='.txt')
    if user_choice == '':
        user_choice = default
    dir_name, file_name = os.path.split(user_choice)
    file_base = file_name.split('.', 1)[0]
    run = RunTools.Run(dir_name, file_base, True)
    viewer = LinkingEvaluation.view_progress
    LinkingNN.begin_training(run, viewer)

if __name__ == '__main__':
    main()
