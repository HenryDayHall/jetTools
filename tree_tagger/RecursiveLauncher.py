""" Module to run the recusive NN """
import os
import threading
from tree_tagger import RecursiveNN, RunTools, InputTools, RecursiveEvaluation
from matplotlib import pyplot as plt


def main():
    plotting = InputTools.yesNo_question("Do you want plots? ")
    tst_dir = "./fakereco"
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
    run = RunTools.RecursiveRun(dir_name, file_base, True, True)

    if plotting:
        training_thread = threading.Thread(target=RecursiveNN.begin_training, args=(run, ))
        training_thread.start()
        live_plot = RunTools.Liveplot(run)

        plt.clf()
        output = RecursiveEvaluation.apply_recursive_net(run)
        RecursiveEvaluation.plot_hist(*output)
        plt.show()
    else:
        RecursiveNN.begin_training(run)

    #viewer = LinkingEvaluation.ResponsePlot(run)
    #LinkingNN.begin_training(run, viewer.update)
    input("What do you think")
    

if __name__ == '__main__':
    main()
