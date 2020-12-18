""" Testing problems with mocking function """

def neigh(list_of_sounds):
    list_of_sounds.append("neigh")


def pet_horse():
    sounds = []
    neigh(sounds)
    return sounds

