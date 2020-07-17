""" Test for the InputTools.py module """
import sys
from contextlib import contextmanager
from tools import TempTestDir
import os
import io
from tree_tagger import InputTools
import unittest.mock
from ipdb import set_trace as st
import numpy.testing as tst


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def fake_tab_complete(possibilities_function, message, previous=None):
    completer_delims = '\t'
    user_input = input(message)
    if user_input.endswith(completer_delims):
        user_input = user_input[:-1]
        try:
            return possibilities_function(user_input, 0)
        except IndexError:
            return user_input
    return user_input


def test_get_file_name():
    with TempTestDir("tst") as dir_name:
        file1 = "meow.cat"
        file2 = "maple.syrup"
        new_name1 = os.path.join(dir_name, file1)
        open(new_name1, 'w').close()  # create the file
        new_name2 = os.path.join(dir_name, file2)
        open(new_name2, 'w').close()  # create the file
        # chcek get file name allows arbitary input
        arbitary = "jflsafhdkas ;lasjdf"
        with replace_stdin(io.StringIO(arbitary)):
            found = InputTools.get_file_name("Msg ")
        assert found.strip() == arbitary
        with replace_stdin(io.StringIO(arbitary)):
            found = InputTools.get_file_name("Msg ", "cat")
        assert found.strip() == arbitary
        # tab completing a file that ends in cat should get the first name
        complete = os.path.join(dir_name, 'm') + '\t'
        with replace_stdin(io.StringIO(complete)):
            with unittest.mock.patch('tree_tagger.InputTools.tab_complete',
                                     new=fake_tab_complete):
                found = InputTools.get_file_name("Msg ", "cat")
        assert found.strip() == new_name1, f"Found {found} from {complete}, expected {new_name1}"





def test_get_dir_name():
    with TempTestDir("tst") as dir_name:
        dir1 = "meow"
        dir2 = "woof"
        file1 = "maple.syrup"
        new_file1 = os.path.join(dir_name, file1)
        open(new_file1, 'w').close()  # create the file
        new_dir1 = os.path.join(dir_name, dir1)
        os.mkdir(new_dir1)
        new_dir2 = os.path.join(dir_name, dir2)
        os.mkdir(new_dir2)
        # chcek get dir name allows arbitary input
        arbitary = "jflsafhdkas ;lasjdf"
        with replace_stdin(io.StringIO(arbitary)):
            found = InputTools.get_dir_name("Msg ")
        assert found.strip() == arbitary
        # tab completing a dir should get the dir back
        complete = os.path.join(dir_name, 'm') + '\t'
        with replace_stdin(io.StringIO(complete)):
            with unittest.mock.patch('tree_tagger.InputTools.tab_complete',
                                     new=fake_tab_complete):
                found = InputTools.get_dir_name("Msg ")
        new_dir1 += '/'
        assert found.strip() == new_dir1, f"Found {found} from {complete}, expected {new_dir1}"


def test_list_complete():
    select_from = ['aaa', 'abb', 'bbb']
    # chcek get list complete allows arbitary input
    arbitary = "jflsafhdkas ;lasjdf"
    with replace_stdin(io.StringIO(arbitary)):
        found = InputTools.list_complete("Msg ", select_from)
    assert found.strip() == arbitary
    # tab completing 
    complete = 'b\t'
    with replace_stdin(io.StringIO(complete)):
        with unittest.mock.patch('tree_tagger.InputTools.tab_complete',
                                 new=fake_tab_complete):
            found = InputTools.list_complete("Msg ", select_from)
    expected = select_from[-1]
    assert found.strip() == expected, f"Found {found} from {complete}, expected {expected}"



def test_yesNo_question():
    truthies = ['y', 'yes', '1', 'true', 'Y', 'YES']
    for inp in truthies:
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.yesNo_question("Msg ")
        assert found
    falsies = ['n', 'no', '0', 'false', 'N', 'No']
    for inp in falsies:
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.yesNo_question("Msg ")
        assert not found


def test_get_literal():
    inputs = ['1', '1.', 'True', '[1,2, 3]', '(3,4)']
    expecteds = [1, 1., True, [1,2, 3], (3,4)]
    for inp, exp in zip(inputs, expecteds):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.get_literal("Msg ")
        assert type(found) == type(exp)
        tst.assert_allclose(found, exp)
    inputs = ['1', '1.', 'True', '[1,2, 3]', '(3,4)']
    expecteds = [1., 1, 1, (1,2, 3), [3,4]]
    converters = [float, int, int, tuple, list]
    for inp, exp, conv in zip(inputs, expecteds, converters):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.get_literal("Msg ", conv)
        assert type(found) == type(exp)
        tst.assert_allclose(found, exp)


def test_print_strlist():
    InputTools.print_strlist(['foo', '3', 'dog'])


def test_get_time():
    hours = 1
    mins = 1
    seconds = 1
    expected = hours*60**2 + mins*60 + seconds
    to_input = f'{hours}\n{mins}\n{seconds}'
    with replace_stdin(io.StringIO(to_input)):
        found = InputTools.get_time("Msg ")
    assert found == expected


def test_select_value():
    default = 4.
    inputs = ['1', '1.', '-2', ' ']
    expecteds = [1., 1., -2., default]
    for inp, exp in zip(inputs, expecteds):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.select_value("dog ", default)
        assert type(found) == type(exp)
        assert found == exp
    inputs = ['1', '1.', ' ']
    expecteds = [1., 1, default]
    converters = [float, int, int]
    for inp, exp, conv in zip(inputs, expecteds, converters):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.select_value("dog ", default, conv)
        assert type(found) == type(exp)
        assert found == exp


def test_select_values():
    default = [4., 12.]
    inputs = ['1 4', '1., 10', '-2', ' ']
    expecteds = [[1., 4.], [1., 10.], [-2.], default]
    for inp, exp in zip(inputs, expecteds):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.select_values("friends", ["dog ", "frog"], default)
        assert isinstance(found[0], type(exp[0]))
        tst.assert_allclose(found, exp)
    inputs = ['1 4', '1.3, 10.2', '-2']
    expecteds = [[1., 4.], [1, 10], [-2]]
    converters = [float, int, int]
    for inp, exp, conv in zip(inputs, expecteds, converters):
        with replace_stdin(io.StringIO(inp)):
            found = InputTools.select_values("animals", ["dog ", "frog"], default, conv)
        tst.assert_allclose(found, exp)


def test_PreSelections():
    pass
