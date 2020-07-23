""" file to store all the constants """

# for all sorts of user inputs
lowerCase_truthies = {'': False, 'n': False, 'false': False,
                      'no': False, '0': False,
                      'y': True, 'true': True, 'yes': True,
                      '1': True}

# for the output of the Tracks Towers linking net
# define a set of match status
CORRECT_MATCH = 1
INCORRECT_MATCH = 2
NO_TOWER = 3

# define some parameters for jet clustering
min_jetpt = 20.
min_ntracks = 2
max_tagangle = 0.8

# soemtiems the coordinates need to go in a grid
coordinate_order = ["Energy", "Px", "Py", "Pz", "PT", "Rapidity", "Phi"]

# for checking if a parameter is valid
numeric_classes = {'nn': 'natrual number', 'pdn': 'positive definite number', 'rn': 'real number'}

def is_numeric_class(param, permitted):
    """
    Check if the given value fits the critera of the
    selected numeric class.

    Parameters
    ----------
    param : object
        python object to be tested
        
    permitted : str
        name of the numeric class, as given in
        the values of numeric_classes

    Returns
    -------
    : bool
        does the param fit the critera of permitted

    """
    if permitted == 'natrual number':
        try:
            int_value = int(param)
            if int_value <= 0:
                return False
            if int_value != param:
                return False
        except (ValueError, TypeError, OverflowError):
            return False
    elif permitted == 'positive definite number':
        try:
            return param > 0
        except TypeError:
            return False
    elif permitted == 'real number':
        try:
            float(param)
        except TypeError:
            return False
    else:
        raise ValueError(f"permitted {permitted} not a known numeric type")
    return True



                
