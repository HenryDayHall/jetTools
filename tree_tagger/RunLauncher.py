""" Module to run the recusive or simple NN """
import os
import threading
import ast
from tree_tagger import RecursiveNN, RunTools, InputTools, RunEvaluation, StandardNN, JetBDT
from matplotlib import pyplot as plt


def main():
    plotting = InputTools.yesNo_question("Do you want progress plots? ")
    tst_dir = InputTools.get_dir_name("Where are the runs written? ")
    in_tst = os.listdir(tst_dir)
    runs_in_tst = sorted([name for name in in_tst
                          if name.startswith("run") and name.endswith(".txt")],
                          key=lambda name: int(name[3:-4]))
    default = os.path.join(tst_dir, runs_in_tst[-1])
    print("Pick a run file;")
    user_choice = InputTools.get_file_name(f"(Enter for {default}) ", file_ending='.txt')
    if user_choice == '':
        user_choice = default
    dir_name, file_name = os.path.split(user_choice)
    file_base = file_name.split('.', 1)[0]
    # get the run type by processing the first line
    with open(user_choice, 'r') as run_file:
        try:
            line1 = run_file.readline()
        except SyntaxError:  # only one line in file, no newline
            line1 = run_file.read()
        line1 = line1.strip().split('"')[1]  # take the quotes off

    settings = ast.literal_eval(line1)
    run_type = settings['net_type']
    print(f"Run type is {run_type}")
    if run_type == 'tracktower_projectors':
        run = RunTools.RecursiveRun(dir_name, file_base, accept_empty=True, writing=plotting)
        train = RecursiveNN.begin_training
    elif run_type == 'standard':
        run = RunTools.FlatJetRun(dir_name, file_base, accept_empty=True, writing=plotting)
        train = StandardNN.begin_training
    elif run_type == 'bdt':
        run = RunTools.SklearnJetRun(dir_name, file_base, accept_empty=True, writing=plotting)
        train = JetBDT.begin_training

    if True:
        run.dataset.write("Standardruns")

    if plotting:
        training_thread = threading.Thread(target=train, args=(run, ))
        training_thread.start()
        live_plot = RunTools.Liveplot(run)

        plt.clf()
        RunEvaluation.plot_hist(run)
        plt.show()
    else:
        train(run)
        plt.clf()
        RunEvaluation.plot_hist(run)
        plt.show()

    input("What do you think")
    

if __name__ == '__main__':
    main()
