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
max_tagangle = 1.
