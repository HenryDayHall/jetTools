""" Module to run the linking NN """
import os
import threading
from jet_tools.tree_tagger import LinkingNN, LinkingEvaluation, RunTools, InputTools
from matplotlib import pyplot as plt


def main():
    """ """
    tst_dir = "./big_ds"
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
    run = RunTools.Run(dir_name, file_base, True, True)

    training_thread = threading.Thread(target=LinkingNN.begin_training, args=(run, ))
    training_thread.start()
    live_plot = RunTools.Liveplot(run)

    plt.clf()
    output = LinkingEvaluation.apply_linking_net(run)
    LinkingEvaluation.plot_distances(output)
    plt.show()

    #viewer = LinkingEvaluation.ResponsePlot(run)
    #LinkingNN.begin_training(run, viewer.update)
    input("What do you think")
    

if __name__ == '__main__':
    main()
