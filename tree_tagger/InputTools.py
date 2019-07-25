# some tools for soliciting user input
from ipdb import set_trace as st
import os
import readline
import glob
from tree_tagger import Constants

def getfilename(message, file_ending=''):
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
        return matching[state]

    filename = tabComplete(path_completer, message)
    return filename

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

    selection = tabComplete(list_completer, message)
    return selection

def tabComplete(possibilities_function, message):
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
    """ Get yes to no from the user.

    Parameters
    ----------
    question : str
        question to display at the prompt

    Returns
    -------
    answer : bool
        the users response
    
    """
    user = input(question).strip().lower()
    while user not in Constants.lowerCase_truthies:
        print("Not a valid answer, please give 'y' or 'n'.")
        return yesNo_question(question)
    return Constants.lowerCase_truthies[user]
