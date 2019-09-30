""" some tools for soliciting user input, works in python 2 and 3 
mostly intrested in autocompletion """
from __future__ import print_function
from ipdb import set_trace as st
import numpy as np
import os
import readline
import glob

try: input = raw_input
except NameError: pass

def get_file_name(message, file_ending=''):
    """ Get a file name from the user, with tab completion.    

    Parameters
    ----------
    message : str
        To be printed at the prompt.
        
    file_ending : str
        (Default value = '')
        To limit the sugestions to a particular file ending.

    Returns
    -------
    file_name : str
        The input from the user (not forced to be a file name)
    
    """
    # file tab completion
    def path_completer(start, state):
        line = readline.get_line_buffer().split()
        matching = [x for x in glob.glob(start+'*'+file_ending)]
        # grab matching directories too
        if os.path.isdir(start): # user gave a dir
            real_dir = start
            user_dir = start
            user_incomplete = ''
        else:  # user gave incomplete dir or other
            user_dir, user_incomplete = os.path.split(start)
            if user_dir in ('', '.', './'):  # current dir
                real_dir = '.'  # needed so os.walk works
            else:  # it's a directory (hopefully...)
                real_dir = user_dir
        dirs_here = next(os.walk(real_dir))[1]
        # go back to the format the user chose
        matching_dirs = [os.path.join(user_dir, name) + '/' for name in dirs_here
                         if name.startswith(user_incomplete)]
        matching += matching_dirs
        return matching[state]

    filename = tab_complete(path_completer, message)
    return filename


def get_dir_name(message):
    """ Get a dir name from the user, with tab completion.    

    Parameters
    ----------
    message : str
        To be printed at the prompt.
        
    Returns
    -------
    dir_name : str
        The input from the user (not forced to be a file name)
    
    """
    # dir tab completion
    def path_completer(start, state):
        line = readline.get_line_buffer().split()
        matching = [x for x in glob.glob(start+'*/')]
        return matching[state]

    dir_name = tab_complete(path_completer, message)
    return dir_name


def list_complete(message, possibilities):
    """ Get a string from the user, with tab completion from list.    

    Parameters
    ----------
    message : str
        To be printed at the prompt.
        
    possibilities : list of str
        Entries that should tab complete

    Returns
    -------
    selection : str
        The input from the user (not forced to be in the list)
    
    """
    def list_completer(text, state):
        line   = readline.get_line_buffer()
        
        if not line:
            return [c + " " for c in possibilities][state]
        
        else:
            return [c + " " for c in possibilities if c.startswith(line)][state]

    selection = tab_complete(list_completer, message)
    return selection


def tab_complete(possibilities_function, message):
    """ Create a tab completion based on a function

    Parameters
    ----------
    possibilities_function : callable
        function that takes two parameters,
        the starting string entered by the user
        and the option number to return
        
    message : str
        message to display to the user at the prompt
        

    Returns
    -------
    user_input : str
        the input from the user, 
        this may not be a tab completion
    
    """
    # file tab completion
    readline.set_completer_delims('\t')
    readline.set_completer(possibilities_function)
    readline.parse_and_bind("tab: complete")
    filename = input(message)
    readline.set_completer()  # remove tab completion
    return filename


def yesNo_question(question):
    """ Get yes or no from the user.

    Parameters
    ----------
    question : str
        question to display at the prompt

    Returns
    -------
    answer : bool
        the users response
    
    """
    lowerCase_answers = {'': False, 'n': False, 'false': False,
                         'no': False, '0': False,
                         'y': True, 'true': True, 'yes': True,
                         '1': True}
    user = input(question).strip().lower()
    while user not in lowerCase_answers:
        print("Not a valid answer, please give 'y' or 'n'.")
        return yesNo_question(question)
    return lowerCase_answers[user]


def print_strlist(my_list):
    rows, _ = os.popen('stty size', 'r').read().split()
    rows = int(rows)
    entry_width = max([len(key)+1 for key in my_list])
    per_row = rows//entry_width + 1
    for i, entry in enumerate(my_list):
        print(entry.ljust(entry_width), end='')
        if i%per_row == 0 and i>0:
            print()
    print()


def get_time(question):
    '''
    Ask the user to give a time in hours minutes and seconds.

    Parameters
    ----------
    question : str
       Message to be displayed
        

    Returns
    -------
    time : int
       time in seconds as specifed by the user

    '''
    print(question)
    hours =   input("enter hours then minutes then seconds|   hours= ")
    minutes = input("                                     |+minutes= ")
    seconds = input("                                     |+seconds= ")
    hours = 0. if hours=='' else float(hours)
    minutes = 0. if minutes=='' else float(minutes)
    seconds = 0. if seconds=='' else float(seconds)
    return 60*60*hours + 60*minutes + seconds

def select_values(pretty_name, column_names, defaults, value_class=np.float):
    print("Select {}".format(pretty_name))
    if column_names is not None:
        print("Format is {}".format(', '.join(column_names)))
        n_columns = len(defaults)
        assert len(defaults) == len(column_names)
    print("Typical {} are {}".format(pretty_name, ' '.join([str(d) for d in defaults])))
    inp = input('{} (enter for typical)= '.format(pretty_name))
    if inp == '':
        chosen = defaults
    else:
        split_inp = inp.replace(',', ' ').split()
        if column_names is None:
            n_columns = len(split_inp)
        chosen = np.array([value_class(i) for i in split_inp[:n_columns]])
    print("{} = [{}]".format(pretty_name, ', '.join([str(c) for c in chosen])))
    return chosen

def select_value(pretty_name, default, value_class=np.float):
    print("Select {}".format(pretty_name))
    print("Typical {} is {}".format(pretty_name, str(default)))
    inp = input('{} (enter for typical)= '.format(pretty_name))
    if inp == '':
        chosen = default
    else:
        chosen = value_class(inp)
    print("{} = {}".format(pretty_name, str(chosen)))
    return chosen
