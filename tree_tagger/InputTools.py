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

up = '\x1b[A'
down = '\x1b[B'

def get_previous(function_name, message):
    """
    

    Parameters
    ----------
    function_name :
        
    message :
        

    Returns
    -------

    """
    delimiter = ','
    message = message.replace(delimiter, '_')
    memory_file = './autofill_memory.csv'
    try:
        existing = np.loadtxt(memory_file, delimiter=delimiter, dtype=str)
        if len(existing) == 0:
            return ''
        existing = existing.reshape((-1, 3))
    except OSError:
        return ''
    match,  = np.where((existing[:, :2] == [function_name, message]).all(axis=1))
    if len(match) == 0:
        return ''
    else:
        return existing[match[0], 2]
    

def set_previous(function_name, message, previous):
    """
    

    Parameters
    ----------
    function_name :
        
    message :
        
    previous :
        

    Returns
    -------

    """
    if previous in ['', up, down]:
        return
    delimiter = ','
    message = message.replace(delimiter, '_')
    memory_file = './autofill_memory.csv'
    try:
        existing = np.loadtxt(memory_file, delimiter=delimiter, dtype=str)
        if len(existing) == 0:
            existing = np.array([[function_name, message, previous]])
        else:
            existing = existing.reshape((-1, 3))
            match = np.where((existing[:, :2] == [function_name, message]).all(axis=1))[0]
            if len(match) == 0:
                existing = np.vstack((existing, [function_name, message, previous]))
            else:
                existing[match[0], 2] = previous
    except OSError:
        existing = np.array([[function_name, message, previous]])
    np.savetxt(memory_file, existing, delimiter=delimiter, header="Memory of user inputs", fmt='%s')


def get_file_name(message, file_ending=''):
    """
    Get a file name from the user, with tab completion.

    Parameters
    ----------
    message : str
        To be printed at the prompt.
    file_ending : str
        (Default value = '')
        To limit the sugestions to a particular file ending.

    Returns
    -------

    
    """
    previous_choice = get_previous("get_file_name", message)
    # file tab completion
    def path_completer(start, state):
        """
        

        Parameters
        ----------
        start :
            
        state :
            

        Returns
        -------

        """
        line = readline.get_line_buffer().split()
        matching = [x for x in glob.glob(start+'*'+file_ending)]
        matching += [x for x in glob.glob(start+'*/')]
        return matching[state]
    filename = tab_complete(path_completer, message, previous_choice)
    set_previous("get_file_name", message, filename)
    return filename


def get_dir_name(message):
    """
    Get a dir name from the user, with tab completion.

    Parameters
    ----------
    message : str
        To be printed at the prompt.

    Returns
    -------

    
    """
    previous_choice = get_previous("get_dir_name", message)
    # dir tab completion
    def path_completer(start, state):
        """
        

        Parameters
        ----------
        start :
            
        state :
            

        Returns
        -------

        """
        line = readline.get_line_buffer().split()
        matching = [x for x in glob.glob(start+'*/')]
        return matching[state]
    dir_name = tab_complete(path_completer, message, previous_choice)
    set_previous("get_dir_name", message, dir_name)
    return dir_name


def list_complete(message, possibilities):
    """
    Get a string from the user, with tab completion from list.

    Parameters
    ----------
    message : str
        To be printed at the prompt.
    possibilities : list of str
        Entries that should tab complete

    Returns
    -------

    
    """
    previous_choice = get_previous("list_complete", message)
    def list_completer(text, state):
        """
        

        Parameters
        ----------
        text :
            
        state :
            

        Returns
        -------

        """
        line = readline.get_line_buffer()
        if not line:
            return [c + " " for c in possibilities][state]
        else:
            return [c + " " for c in possibilities if c.startswith(line)][state]
    selection = tab_complete(list_completer, message, previous_choice)
    set_previous("list_complete", message, selection)
    return selection


def tab_complete(possibilities_function, message, previous=None):
    """
    Create a tab completion based on a function

    Parameters
    ----------
    possibilities_function : callable
        function that takes two parameters,
        the starting string entered by the user
        and the option number to return
    message : str
        message to display to the user at the prompt
    previous :
         (Default value = None)

    Returns
    -------

    
    """
    # file tab completion
    readline.set_completer_delims('\t')
    readline.set_completer(possibilities_function)
    readline.parse_and_bind("tab: complete")
    if previous not in [None, '']:
        readline.add_history(previous)
    user_input = input(message)
    readline.set_completer()  # remove tab completion
    return user_input


def yesNo_question(question):
    """
    Get yes or no from the user.

    Parameters
    ----------
    question : str
        question to display at the prompt

    Returns
    -------

    
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
    """
    

    Parameters
    ----------
    my_list :
        

    Returns
    -------

    """
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
    """
    Ask the user to give a time in hours minutes and seconds.

    Parameters
    ----------
    question : str
        Message to be displayed

    Returns
    -------

    
    """
    print(question)
    hours =   input("enter hours then minutes then seconds|   hours= ")
    minutes = input("                                     |+minutes= ")
    seconds = input("                                     |+seconds= ")
    hours = 0. if hours=='' else float(hours)
    minutes = 0. if minutes=='' else float(minutes)
    seconds = 0. if seconds=='' else float(seconds)
    return 60*60*hours + 60*minutes + seconds


def select_values(pretty_name, column_names, defaults, value_class=np.float):
    """
    

    Parameters
    ----------
    pretty_name :
        
    column_names :
        
    defaults :
        
    value_class :
         (Default value = np.float)

    Returns
    -------

    """
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
    """
    

    Parameters
    ----------
    pretty_name :
        
    default :
        
    value_class :
         (Default value = np.float)

    Returns
    -------

    """
    print("Select {}".format(pretty_name))
    print("Typical {} is {}".format(pretty_name, str(default)))
    inp = input('{} (enter for typical)= '.format(pretty_name))
    if inp == '':
        chosen = default
    else:
        chosen = value_class(inp)
    print("{} = {}".format(pretty_name, str(chosen)))
    return chosen
