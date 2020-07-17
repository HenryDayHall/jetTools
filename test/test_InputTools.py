""" Test for the InputTools.py module """
import sys
from contextlib import contextmanager
from tools import TempTestDir
import os
import io
from tree_tagger import InputTools
import unittest.mock


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
    pass


def test_yesNo_question():
    pass


def test_get_literal():
    pass


def test_print_strlist():
    pass


def test_get_time():
    pass


def test_select_values():
    pass


def test_select_value():
    pass


def test_PreSelections():
    pass
