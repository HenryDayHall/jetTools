""" some tools for soliciting user input, works in python 2 and 3 
mostly intrested in autocompletion """
from __future__ import print_function
from ipdb import set_trace as st
import numpy as np
import os
import readline
import glob
import ast
import re

try: input = raw_input
except NameError: pass

up = '\x1b[A'
down = '\x1b[B'

# a dictionary of a preselected reponse per quesiton

class PreSelections:
    """ """
    _sep = '||'
    def __init__(self, file_name=None):
        self.questions = []
        self.consistant_length = []
        self.answers = []
        self.question_reached = 0
        if file_name is not None:
            with open(file_name, 'r') as in_file:
                str_lines = in_file.readlines()
                self.questions = [q.replace('\n', '') for q in str_lines[0].split(self._sep)]
                self.consistant_length = [int(q) for q in str_lines[1].split(self._sep)]
                # answers may contain numpy arrays,
                for answer in str_lines[2].split(self._sep):
                    if answer.endswith('\n'):
                        answer = answer[:-1]
                    answer = answer.strip()
                    try:
                        answer = ast.literal_eval(answer)
                    except (SyntaxError, ValueError):
                        if ' ' in answer:
                            answer = [ast.literal_eval(a) for a in answer[1:-1].split()]
                    self.answers.append(answer)
            assert len(self.questions) == len(self.consistant_length)
            assert len(self.questions) == len(self.answers)

    def set(self, questions=None, varation=None, answers=None):
        """
        

        Parameters
        ----------
        questions :
             (Default value = None)
        varation :
             (Default value = None)
        answers :
             (Default value = None)

        Returns
        -------

        """
        if questions is None:
            assert varation is None and answers is None
            questions = []
            consistant_length = []
            answers = []
        self.questions = []
        self.consistant_length = []
        self.answers = answers

    def __getitem__(self, message):
        if self.question_reached >= len(self.questions):
            return None
        response = None
        consistant_length = self.consistant_length[self.question_reached]
        if consistant_length > 0:
            message = message[:consistant_length]
            stored_message = self.questions[self.question_reached][:consistant_length]
        else:
            stored_message = self.questions[self.question_reached]
        if message == stored_message:
            response = self.answers[self.question_reached]
            print(f"AUTO: {message} -> {response}")
            self.question_reached += 1
        else:
            st()
        return response

    def append(self, message, response, consistant_length=-1):
        """
        

        Parameters
        ----------
        message :
            
        response :
            
        consistant_length :
             (Default value = -1)

        Returns
        -------

        """
        self.questions.append(message)
        self.consistant_length.append(consistant_length)
        self.answers.append(response)

    def write(self, file_name):
        """
        

        Parameters
        ----------
        file_name :
            

        Returns
        -------

        """
        lines = [self._sep.join([str(p) for p in line]) for line in
                 [self.questions, self.consistant_length, self.answers]]
        with open(file_name, 'w') as out:
            out.writelines(os.linesep.join(lines))

    def replace_string(self, old, new):
        """
        

        Parameters
        ----------
        old :
            
        new :
            

        Returns
        -------

        """
        for i, answer in enumerate(self.answers):
            if isinstance(answer, str):
                self.answers[i] = re.sub(old, new, answer)

    def replace_all_answers(self, question_start, new):
        """
        

        Parameters
        ----------
        question_start :
            
        new :
            

        Returns
        -------

        """
        for i, question in enumerate(self.questions):
            if question.startswith(question_start):
                self.answers[i] = new


pre_selections = PreSelections()
last_selections = PreSelections()

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


def get_file_name(message, file_ending='', consistant_length=-1):
    """
    Get a file name from the user, with tab completion.

    Parameters
    ----------
    message : str
        To be printed at the prompt.
    file_ending : str
        (Default value = '')
        To limit the sugestions to a particular file ending.
    consistant_length :
         (Default value = -1)

    Returns
    -------
    file_name : str
        The input from the user (not forced to be a file name)

    
    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
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
    last_selections.append(message, filename, consistant_length)
    return filename


def get_dir_name(message, consistant_length=-1):
    """
    Get a dir name from the user, with tab completion.

    Parameters
    ----------
    message : str
        To be printed at the prompt.
    consistant_length :
         (Default value = -1)

    Returns
    -------
    dir_name : str
        The input from the user (not forced to be a file name)

    
    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
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
    last_selections.append(message, dir_name, consistant_length)
    return dir_name


def list_complete(message, possibilities, consistant_length=-1):
    """
    Get a string from the user, with tab completion from list.

    Parameters
    ----------
    message : str
        To be printed at the prompt.
    possibilities : list of str
        Entries that should tab complete
    consistant_length :
         (Default value = -1)

    Returns
    -------
    selection : str
        The input from the user (not forced to be in the list)

    
    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
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
        #line = readline.get_line_buffer()
        #if not line:
        #    return [c + " " for c in possibilities][state]
        #else:
        #    return [c + " " for c in possibilities if c.startswith(line)][state]
        if not text:
            return [c + " " for c in possibilities][state]
        else:
            return [c + " " for c in possibilities if c.startswith(text)][state]
    selection = tab_complete(list_completer, message, previous_choice)
    set_previous("list_complete", message, selection)
    last_selections.append(message, selection, consistant_length)
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
    user_input : str
        the input from the user,
        this may not be a tab completion

    
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


def yesNo_question(message, consistant_length=-1):
    """
    Get yes or no from the user.

    Parameters
    ----------
    message : str
        message to display at the prompt
    consistant_length :
         (Default value = -1)

    Returns
    -------
    answer : bool
        the users response

    
    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
    lowerCase_answers = {'': False, 'n': False, 'false': False,
                         'no': False, '0': False,
                         'y': True, 'true': True, 'yes': True,
                         '1': True}
    user = input(message).strip().lower()
    while user not in lowerCase_answers:
        print("Not a valid answer, please give 'y' or 'n'.")
        return yesNo_question(message)
    selection = lowerCase_answers[user]
    last_selections.append(message, selection, consistant_length)
    return selection


def get_literal(message, converter=None, consistant_length=-1):
    """
    

    Parameters
    ----------
    message :
        
    converter :
         (Default value = None)
    consistant_length :
         (Default value = -1)

    Returns
    -------

    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
    try:
        user = input(message)
        response = ast.literal_eval(user)
        if converter is not None:
            response = converter(response)
    except ValueError:
        print(f"'{user}' is not a valid input. Try again.")
        response = get_literal(message, converter)
    last_selections.append(message, response, consistant_length)
    return response


def print_strlist(my_list):
    """
    

    Parameters
    ----------
    my_list :
        

    Returns
    -------

    """
    try:
        cols, _ = os.popen('stty size', 'r').read().split()
    except ValueError:  # sometimes fails
        import shutil
        fall_back = (80, 20)
        cols, _ = shutil.get_terminal_size(fall_back)
    cols = int(cols)
    entry_width = max([len(key)+1 for key in my_list])
    per_row = cols//entry_width + 1
    for i, entry in enumerate(my_list):
        print(entry.ljust(entry_width), end='')
        if i%per_row == 0 and i>0:
            print()
    print()


def get_time(message, consistant_length=-1):
    """
    Ask the user to give a time in hours minutes and seconds.

    Parameters
    ----------
    message : str
        Message to be displayed
    consistant_length :
         (Default value = -1)

    Returns
    -------
    time : int
        time in seconds as specifed by the user

    
    """
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
    print(message)
    hours =   input("enter hours then minutes then seconds|   hours= ")
    minutes = input("                                     |+minutes= ")
    seconds = input("                                     |+seconds= ")
    hours = 0. if hours=='' else float(hours)
    minutes = 0. if minutes=='' else float(minutes)
    seconds = 0. if seconds=='' else float(seconds)
    response =  60*60*hours + 60*minutes + seconds
    last_selections.append(message, response, consistant_length)
    return response


def select_values(pretty_name, column_names, defaults, value_class=np.float, consistant_length=-1):
    """
    

    Parameters
    ----------
    pretty_name :
        
    column_names :
        
    defaults :
        
    value_class :
         (Default value = np.float)
    consistant_length :
         (Default value = -1)

    Returns
    -------

    """
    # make the message
    message = f"Select {pretty_name}\n"
    if column_names is not None:
        message += "Format is {}\n".format(', '.join(column_names))
        n_columns = len(defaults)
        assert len(defaults) == len(column_names)
    message += "Typical {} are {}\n".format(pretty_name, ' '.join([str(d) for d in defaults]))
    message += f'{pretty_name} (enter for typical)= '
    # check for an existing rsponse
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
    inp = input(message)
    if inp == '':
        chosen = defaults
    else:
        split_inp = inp.replace(',', ' ').split()
        if column_names is None:
            n_columns = len(split_inp)
        chosen = np.array([value_class(i) for i in split_inp[:n_columns]])
    print("{} = [{}]".format(pretty_name, ', '.join([str(c) for c in chosen])))
    last_selections.append(message, chosen, consistant_length)
    return chosen


def select_value(pretty_name, default, value_class=np.float, consistant_length=-1):
    """
    

    Parameters
    ----------
    pretty_name :
        
    default :
        
    value_class :
         (Default value = np.float)
    consistant_length :
         (Default value = -1)

    Returns
    -------

    """
    message = f"Select {pretty_name}\n"
    message += f"Typical {pretty_name} is {default}\n"
    message += f'{pretty_name} (enter for typical)= '
    # check for an existing rsponse
    prepared_response = pre_selections[message]
    if prepared_response is not None:
        return prepared_response
    inp = input(message).strip()
    if inp == '':
        chosen = default
    else:
        try:
            chosen = value_class(inp)
        except ValueError:
            chosen = value_class(ast.literal_eval(inp))
    print("{} = {}".format(pretty_name, str(chosen)))
    last_selections.append(message, chosen, consistant_length)
    return chosen

