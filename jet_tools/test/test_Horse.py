""" Testing problems with mocking functions"""
from jet_tools.tree_tagger import Horse
import unittest

def fake_purr(list_of_sounds):
    list_of_sounds.append("meow")

def purr(list_of_sounds):
    list_of_sounds.append("purr")


def pet_cat():
    sounds_heard = []
    purr(sounds_heard)
    return sounds_heard


def test_pet_cat():
    from jet_tools.test import test_Horse
    with unittest.mock.patch('jet_tools.test.test_Horse.purr', new=fake_purr):
        sound = test_Horse.pet_cat()
    assert sound[0] == 'meow'


def fake_neigh(list_of_sounds):
    list_of_sounds.append("clip clop")


def test_pet_horse():
    with unittest.mock.patch('jet_tools.tree_tagger.Horse.neigh', new=fake_neigh):
        sounds = Horse.pet_horse()
    assert sounds[0] == "clip clop"
