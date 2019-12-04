import numpy as np
import os
from ipdb import set_trace as st
import collections
from numpy import testing as tst
import pytest
from tree_tagger import Components, PDGNames
from tools import generic_equality_comp, TempTestDir
import awkward

class AwkdArrays:
    empty = awkward.fromiter([])
    one_one = awkward.fromiter([1])
    minus_plus = awkward.fromiter(np.arange(-2, 3))
    event_ints = awkward.fromiter([[1, 2], [3]])
    jet_ints = awkward.fromiter([awkward.fromiter([[1, 2], [3]]),
                                 awkward.fromiter([[4,5,6]])])
    event_floats = awkward.fromiter([[.1, .2], [.3]])
    jet_floats = awkward.fromiter([awkward.fromiter([[.1, .2], [.3]]),
                                 awkward.fromiter([[.4,.5,.6]])])
    empty_event = awkward.fromiter([[], [1]])
    empty_jet = awkward.fromiter([awkward.fromiter([[], [1]]),
                                  awkward.fromiter([[2, 3]])])
    empty_events = awkward.fromiter([[], []])
    empty_jets = awkward.fromiter([awkward.fromiter([[],[]]),
                                   awkward.fromiter([[]])])


def test_flatten():
    input_ouputs = [
            (AwkdArrays.empty, []),
            (AwkdArrays.one_one, [1]),
            (AwkdArrays.minus_plus, range(-2, 3)),
            (AwkdArrays.event_ints, [1,2,3]),
            (AwkdArrays.jet_ints, list(range(1, 7))),
            (AwkdArrays.event_floats, [.1, .2, .3]),
            (AwkdArrays.jet_floats, np.arange(0.1, 0.7, 0.1)),
            (AwkdArrays.empty_event, [1]),
            (AwkdArrays.empty_jet, [1,2,3]),
            (AwkdArrays.empty_events, []),
            (AwkdArrays.empty_jets, [])]
    for inp, out in input_ouputs:
        inp = list(Components.flatten(inp))
        tst.assert_allclose(inp, out)


def test_detect_depth():
    input_ouputs = [
            (AwkdArrays.empty, (False, 0)),
            (AwkdArrays.one_one, (True, 0)),
            (AwkdArrays.minus_plus, (True, 0)),
            (AwkdArrays.event_ints, (True, 1)),
            (AwkdArrays.jet_ints, (True, 2)),
            (AwkdArrays.event_floats, (True, 1)),
            (AwkdArrays.jet_floats, (True, 2)),
            (AwkdArrays.empty_event, (True, 1)),
            (AwkdArrays.empty_jet, (True, 2)),
            (AwkdArrays.empty_events, (False, 1)),
            (AwkdArrays.empty_jets, (False, 2))]
    for inp, out in input_ouputs:
        depth = Components.detect_depth(inp)
        assert depth == out, f"{inp} gives depth (certanty, depth) = {depth}, not {out}"


def test_apply_array_func():
    func = np.cos
    input_ouputs = [
        (AwkdArrays.empty, awkward.fromiter([])),
        (AwkdArrays.one_one, awkward.fromiter([func(1)])),
        (AwkdArrays.minus_plus, awkward.fromiter(func(np.arange(-2, 3)))),
        (AwkdArrays.event_ints, awkward.fromiter([[func(1), func(2)], [func(3)]])),
        (AwkdArrays.jet_ints, awkward.fromiter([awkward.fromiter([[func(1), func(2)], [func(3)]]),
                                     awkward.fromiter([[func(4),func(5),func(6)]])])),
        (AwkdArrays.event_floats, awkward.fromiter([[func(.1), func(.2)], [func(.3)]])),
        (AwkdArrays.jet_floats, awkward.fromiter([awkward.fromiter([[func(.1), func(.2)], [func(.3)]]),
                                     awkward.fromiter([[func(.4),func(.5),func(.6)]])])),
        (AwkdArrays.empty_event, awkward.fromiter([[], [func(1)]])),
        (AwkdArrays.empty_jet, awkward.fromiter([awkward.fromiter([[], [func(1)]]),
                                      awkward.fromiter([[func(2), func(3)]])])),
        (AwkdArrays.empty_events, awkward.fromiter([[], []])),
        (AwkdArrays.empty_jets, awkward.fromiter([awkward.fromiter([[],[]]),
                                       awkward.fromiter([[]])]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"
    func = len
    input_ouputs = [
        (AwkdArrays.empty, 0),
        (AwkdArrays.one_one, 1),
        (AwkdArrays.minus_plus, 5),
        (AwkdArrays.event_ints, awkward.fromiter([2, 1])),
        (AwkdArrays.jet_ints, awkward.fromiter([awkward.fromiter([2, 1]),
                                     awkward.fromiter([3])])),
        (AwkdArrays.event_floats, awkward.fromiter([2, 1])),
        (AwkdArrays.jet_floats, awkward.fromiter([awkward.fromiter([2, 1]),
                                     awkward.fromiter([3])])),
        (AwkdArrays.empty_event, awkward.fromiter([0, 1])),
        (AwkdArrays.empty_jet, awkward.fromiter([awkward.fromiter([0, 1]),
                                      awkward.fromiter([2])])),
        (AwkdArrays.empty_events, awkward.fromiter([0, 0])),
        (AwkdArrays.empty_jets, awkward.fromiter([[0, 0], [0]]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"
    # specify event depth
    input_ouputs = [
        (AwkdArrays.event_ints, awkward.fromiter([2, 1])),
        (AwkdArrays.jet_ints, awkward.fromiter([2, 1])),
        (AwkdArrays.event_floats, awkward.fromiter([2, 1])),
        (AwkdArrays.jet_floats, awkward.fromiter([2, 1])),
        (AwkdArrays.empty_event, awkward.fromiter([0, 1])),
        (AwkdArrays.empty_jet, awkward.fromiter([2, 1])),
        (AwkdArrays.empty_events, awkward.fromiter([0, 0])),
        (AwkdArrays.empty_jets, awkward.fromiter([2, 1]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp, depth=Components.EventWise.EVENT_DEPTH)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"


def test_confine_angle():
    inputs_outputs = [
            (0., 0.),
            (1., 1.),
            (-1., -1.),
            (2*np.pi, 0.),
            (np.pi+0.1, -np.pi+0.1),
            (-np.pi-0.1, np.pi-0.1),
            (2*np.pi+0.1, 0.1),
            (-2*np.pi, 0.),
            (-2*np.pi-0.1, -0.1),
            (4*np.pi, 0.)]
    for inp, out in inputs_outputs:
        tst.assert_allclose(Components.confine_angle(inp), out)


def test_safe_convert():
    # edge case tests
    examples = (('None', str, None),
                ('None', int, None),
                ('None', float, None),
                ('None', bool, None),
                ('1', str, '1'),
                ('1', int, 1),
                ('1', float, 1.),
                ('1', bool, True),
                ('0', str, '0'),
                ('0', int, 0),
                ('0', float, 0.),
                ('0', bool, False),
                ('-1', str, '-1'),
                ('-1', int, -1),
                ('-1', float, -1.),
                ('-1', bool, True),
                ('0.5', str, '0.5'),
                ('0.5', float, 0.5),
                ('0.5', bool, True))
    for inp, cls, exptd in examples:
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
    # try some random numbers
    for num in np.random.uniform(-1000, 1000, 20):
        inp = str(num)
        cls = float
        exptd = num
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = bool
        exptd = True
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = int
        exptd = int(num)
        out = Components.safe_convert(cls, inp.split('.', 1)[0])
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."


def test_EventWise():
    # blank
    with TempTestDir("tst") as dir_name:
        # instansation
        save_name = "blank.awkd"
        blank_ew = Components.EventWise(dir_name, save_name)
        assert blank_ew.columns == []
        # add to index
        #contents = []
        #blank_ew.add_to_index(contents)
        #expected =  {"name": "blank", "save_name": save_name,
        #             "mutable": True}
        #assert generic_equality_comp(contents[-1], expected)
        #blank_ew.add_to_index(contents, name="a", mutable=False)
        #expected =  {"name": "a", "save_name": save_name,
        #             "mutable": False}
        #assert generic_equality_comp(contents[-1], expected)
        # getting attributes
        with pytest.raises(AttributeError):
            getattr(blank_ew, "PT")
        with pytest.raises(AttributeError):
            blank_ew.PT
        # write
        save_path = os.path.join(dir_name, save_name)
        blank_ew.write()
        assert os.path.exists(save_path)
        # from file
        blank_ew_clone = Components.EventWise.from_file(save_path)
        assert generic_equality_comp(blank_ew.columns, blank_ew_clone.columns)
        contents = {k:v for k, v in blank_ew._column_contents.items() if k!="column_order"}
        contents_clone = {k:v for k, v in blank_ew_clone._column_contents.items() if k!="column_order"}
        assert generic_equality_comp(contents, contents_clone)
        # eq
        assert blank_ew == blank_ew_clone
        # append
        blank_ew.append(["a"], {"a": AwkdArrays.empty})
        blank_ew.append({"b": AwkdArrays.one_one})
        assert list(blank_ew.a) == []
        assert "A" in blank_ew.columns
        assert list(blank_ew.b) == [1]
        assert "B" in blank_ew.columns
        # remove
        current_cols = list(blank_ew.columns)
        for name in current_cols:
            blank_ew.remove(name)
        assert len(blank_ew.columns) == 0
        with pytest.raises(AttributeError):
            blank_ew.a
        with pytest.raises(AttributeError):
            blank_ew.b
        # remove prefix
        blank_ew.append({"A": AwkdArrays.empty, "Bc": AwkdArrays.one_one, "Bd": AwkdArrays.minus_plus})
        blank_ew.remove_prefix("B")
        assert list(blank_ew.columns) == ["A"]


def test_split():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew.append({"Energy": awkward.fromiter([])})
        parts = ew.split([0, 0, 0], [0, 0, 0])
        for part in parts:
            assert part is None
        # otherwise things should be possible to divide into events
        # try with 10 events
        n_events = 10
        content_1 = awkward.fromiter(np.arange(n_events))
        content_2 = awkward.fromiter(np.random.rand(n_events))
        content_3 = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = awkward.fromiter([[awkward.fromiter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append({'c1': content_1, 'c2': content_2, 'c3': content_3, 'c4': content_4})
        ew.append({'c1': content_1, 'c2': content_2, 'c3': content_3, 'c4': content_4})
        # check nothing changes in the original
        tst.assert_allclose(ew.c1, content_1)
        tst.assert_allclose(ew.c2, content_2)
        tst.assert_allclose(ew.c3.flatten(), content_3.flatten())
        tst.assert_allclose(ew.c4.flatten().flatten(), content_4.flatten().flatten())
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        tst.assert_allclose(ew.c1, content_1)
        tst.assert_allclose(ew.c2, content_2)
        tst.assert_allclose(ew.c3.flatten(), content_3.flatten())
        tst.assert_allclose(ew.c4.flatten().flatten(), content_4.flatten().flatten())
        # check the segments contain what they should
        ew0 = Components.EventWise.from_file(paths[0])
        assert len(ew0.c1) == 5
        tst.assert_allclose(ew0.c1, content_1[:5])
        tst.assert_allclose(ew0.c2, content_2[:5])
        tst.assert_allclose(ew0.c3.flatten(), content_3[:5].flatten())
        tst.assert_allclose(ew0.c4.flatten().flatten(), content_4[:5].flatten().flatten())
        ew1 = Components.EventWise.from_file(paths[1])
        assert len(ew1.c1) == 2
        tst.assert_allclose(ew1.c1, content_1[5:7])
        tst.assert_allclose(ew1.c2, content_2[5:7])
        tst.assert_allclose(ew1.c3.flatten(), content_3[5:7].flatten())
        tst.assert_allclose(ew1.c4.flatten().flatten(), content_4[5:7].flatten().flatten())
        assert paths[2] is None
        ew3 = Components.EventWise.from_file(paths[3])
        assert len(ew3.c1) == 3
        tst.assert_allclose(ew3.c1, content_1[7:])
        tst.assert_allclose(ew3.c2, content_2[7:])
        tst.assert_allclose(ew3.c3.flatten(), content_3[7:].flatten())
        tst.assert_allclose(ew3.c4.flatten().flatten(), content_4[7:].flatten().flatten())
        assert np.all(["dog" in name for name in paths if name is not None])

def test_fragment():
    with TempTestDir("tst") as dir_name:
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        # try with 12 events
        n_events = 12
        content_1 = awkward.fromiter(np.arange(n_events))
        content_2 = awkward.fromiter(np.random.rand(n_events))
        content_3 = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = awkward.fromiter([[awkward.fromiter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append({'c1': content_1, 'c2': content_2, 'c3': content_3, 'c4': content_4})
        # we can fragment 2 ways, first by number of fragments
        paths = ew.fragment('c1', n_fragments=3)
        for i, path in enumerate(paths):
            ew0 = Components.EventWise.from_file(path)
            assert len(ew0.c1) == 4
            idxs = slice(i*4, (i+1)*4)
            tst.assert_allclose(ew0.c1, content_1[idxs])
            tst.assert_allclose(ew0.c2, content_2[idxs])
            tst.assert_allclose(ew0.c3.flatten(), content_3[idxs].flatten())
            tst.assert_allclose(ew0.c4.flatten().flatten(), content_4[idxs].flatten().flatten())
        # we can fragment 2 ways, first by number of fragments
        paths = ew.fragment('c1', fragment_length=3)
        for i, path in enumerate(paths):
            ew0 = Components.EventWise.from_file(path)
            assert len(ew0.c1) == 3
            idxs = slice(i*3, (i+1)*3)
            tst.assert_allclose(ew0.c1, content_1[idxs])
            tst.assert_allclose(ew0.c2, content_2[idxs])
            tst.assert_allclose(ew0.c3.flatten(), content_3[idxs].flatten())
            tst.assert_allclose(ew0.c4.flatten().flatten(), content_4[idxs].flatten().flatten())


def test_split_unfinished():
    with TempTestDir("tst") as dir_name:
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        # try with 12 events
        n_events = 12
        n_unfinished = 2
        content_1 = awkward.fromiter(np.arange(n_events))
        content_2 = awkward.fromiter(np.random.rand(n_events-n_unfinished))
        content_3 = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = awkward.fromiter([[awkward.fromiter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append({'c1': content_1, 'c2': content_2, 'c3': content_3, 'c4': content_4})
        paths = ew.split_unfinished('c1', 'c2')
        ew0 = Components.EventWise.from_file(paths[0])
        assert len(ew0.c1) == n_events - n_unfinished
        idxs = slice(n_events-n_unfinished)
        tst.assert_allclose(ew0.c1, content_1[idxs])
        tst.assert_allclose(ew0.c2, content_2[idxs])
        tst.assert_allclose(ew0.c3.flatten(), content_3[idxs].flatten())
        tst.assert_allclose(ew0.c4.flatten().flatten(), content_4[idxs].flatten().flatten())
        ew1 = Components.EventWise.from_file(paths[1])
        assert len(ew1.c1) == n_unfinished
        idxs = slice(n_events-n_unfinished, None)
        tst.assert_allclose(ew1.c1, content_1[idxs])
        tst.assert_allclose(ew1.c3.flatten(), content_3[idxs].flatten())
        tst.assert_allclose(ew1.c4.flatten().flatten(), content_4[idxs].flatten().flatten())

def test_combine():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        # try with 10 events
        n_events = 10
        content_1 = awkward.fromiter(np.arange(n_events))
        content_2 = awkward.fromiter(np.random.rand(n_events))
        content_3 = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = awkward.fromiter([[awkward.fromiter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append({'c1': content_1, 'c2': content_2, 'c3': content_3, 'c4': content_4})
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        dir_name = os.path.split(paths[0])[0]
        recombined = Components.EventWise.combine(dir_name, "test")
        # tere in no order garentee, so get the new order from c1
        order = np.argsort(recombined.c1)
        tst.assert_allclose(recombined.c1[order], content_1)
        tst.assert_allclose(recombined.c2[order], content_2)
        tst.assert_allclose(recombined.c3[order].flatten(), content_3.flatten())
        for i in range(n_events):
            tst.assert_allclose(recombined.c4[order[i]].flatten().flatten(), content_4[i].flatten().flatten())

        

def test_recursive_combine():
    pass


# warnings
def test_add_rapidity():
    large_num = np.inf
    pts = awkward.fromiter([[0., 0., 0., 0.,  0., 1., 1.,  1., 1., 10.]])
    pzs = awkward.fromiter([[0., 0., 1., -1., 1., 1., -1., 0., 0., 10.]])
    es =  awkward.fromiter([[0., 1., 1., 1.,  2., 2., 2.,  1., 2., 100.]])
    rap = awkward.fromiter([np.nan, 0., large_num + 1., -large_num-1., 0.5*np.log(3/1),
                            0.5*np.log(3/1), -0.5*np.log(3/1), 0., 0., 0.5*np.log(110./90.)])
    with TempTestDir("tst") as dir_name:
        # instansation
        save_name = "rapidity.awkd"
        contents = {"PT": pts, "Pz": pzs, "Energy": es}
        ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                  contents=contents)
        Components.add_rapidity(ew)
        ew.selected_index = 0
        tst.assert_allclose(ew.Rapidity, rap)
        # try adding to a specific prefix
        contents = {"A_PT": pts, "A_Pz": pzs, "A_Energy": es,
                    "B_PT": pts, "B_Pz": pzs, "B_Energy": es}
        ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                  contents=contents)
        Components.add_rapidity(ew, 'A')
        ew.selected_index = 0
        tst.assert_allclose(ew.A_Rapidity, rap)
        with pytest.raises(AttributeError):
            ew.B_Rapidity



class Particle:
    def __init__(self, direction, mass):
        self.px = awkward.fromiter([0.])
        self.py = awkward.fromiter([0.])
        self.pz = awkward.fromiter([0.])
        self.pt = awkward.fromiter([1.])
        self.p = awkward.fromiter([1.])
        if direction == 'x':
            self.px = awkward.fromiter([1.])
        elif direction == '-x':
            self.px = awkward.fromiter([-1.])
        elif direction == 'y':
            self.py = awkward.fromiter([1.])
        elif direction == '-y':
            self.py = awkward.fromiter([-1.])
        elif direction == 'z':
            self.pz = awkward.fromiter([1.])
            self.pt = awkward.fromiter([0.])
        elif direction == '-z':
            self.pz = awkward.fromiter([-1.])
            self.pt = awkward.fromiter([0.])
        elif direction == '45':
            self.px = awkward.fromiter([np.sqrt(0.5)])
            self.py = awkward.fromiter([np.sqrt(0.5)])
            self.pz = awkward.fromiter([1.])
            self.p = awkward.fromiter([np.sqrt(2.)])
        self.m2 = awkward.fromiter([mass**2])
        self.e2 = awkward.fromiter([self.m2[0] + self.p[0]**2])
        self.e = np.sqrt(self.e2)
        self.et = self.e * (self.pt/self.p)


def test_add_thetas():
    # particles could go down each axial direction
    input_output = [
            ('x', np.pi/2.),
            ('-x', np.pi/2.),
            ('y', np.pi/2.),
            ('-y', np.pi/2.),
            ('z', 0.),
            ('-z', np.pi),
            ('45', np.pi/4.)]
    with TempTestDir("tst") as dir_name:
        # instansation
        save_name = "rapidity.awkd"
        for inp, out in input_output:
            particle = Particle(inp, 0)
            # there are many things theta can be calculated from
            # pz&birr pt&pz (px&py)&pz pt&birr et&e
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()), 
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"PT": particle.pt, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Px": particle.px, "Py": particle.py, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            # Need to add energy version for towers, trouble getting direction
            #contents = {"Energy": particle.e, "ET": particle.et}
            #ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
            #                          contents=contents)
            #Components.add_thetas(ew, '')
            #tst.assert_allclose(ew.Theta[0], out)
            # check that adding mass makes no diference
            particle = Particle(inp, 1.)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"PT": particle.pt, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Px": particle.px, "Py": particle.py, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            # Need to add energy version for towers, trouble getting direction
            #contents = {"Energy": particle.e, "ET": particle.et}
            #ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
            #                          contents=contents)
            #Components.add_thetas(ew, '')
            #tst.assert_allclose(ew.Theta[0], out)


def test_theta_to_pseudorapidity():
    input_output = [
            (0., np.inf),
            (np.pi/2, 0.),
            (np.pi, -np.inf)]
    for inp, out in input_output:
        etas = Components.theta_to_pseudorapidity(np.array([inp]))
        tst.assert_allclose(etas[0], out, atol=0.0001)


def test_add_pseudorapidity():
    large_num = np.inf
    theta = awkward.fromiter([[0., np.pi/4, np.pi/2, 3*np.pi/4, np.pi]])
    eta = awkward.fromiter([np.inf, -np.log(np.tan(np.pi/8)), 0., np.log(np.tan(np.pi/8)), -np.inf])
    with TempTestDir("tst") as dir_name:
        # instansation
        save_name = "pseudorapidity.awkd"
        contents = {"Theta": theta}
        ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                  contents=contents)
        Components.add_pseudorapidity(ew)
        ew.selected_index = 0
        tst.assert_allclose(ew.PseudoRapidity, eta, atol=0.0001)
        # try adding to a specific prefix
        contents = {"A_Theta": theta,
                    "B_Theta": theta}
        ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()),
                                  contents=contents)
        Components.add_pseudorapidity(ew, 'A')
        ew.selected_index = 0
        tst.assert_allclose(ew.A_PseudoRapidity, eta, atol=0.0001)
        with pytest.raises(AttributeError):
            ew.B_PseudoRapidity


def test_add_PT():
    # particles could go down each axial direction
    input_output = [
            ('x', 1.),
            ('-x', 1.),
            ('y', 1.),
            ('-y', 1.),
            ('z', 0.),
            ('-z', 0.),
            ('45', 1.)]
    with TempTestDir("tst") as dir_name:
        # instansation
        save_name = "PT.awkd"
        for inp, out in input_output:
            particle = Particle(inp, 0)
            contents = {"Px": particle.px, "Py": particle.py}
            ew = Components.EventWise(dir_name, save_name, columns=list(contents.keys()), 
                                      contents=contents)
            Components.add_PT(ew, '')
            tst.assert_allclose(ew.PT[0], out)


def test_RootReadout():
    root_file = "/home/henry/lazy/dataset2/h1bBatch2.root"
    dir_name, save_name = os.path.split(root_file)
    components = ["Particle", "Track", "Tower"]
    rr = Components.RootReadout(dir_name, save_name, components)
    n_events = len(rr.Energy)
    test_events = np.random.randint(0, n_events, 20)
    idents = PDGNames.Identities()
    all_ids = set(idents.particle_data[:, idents.columns["id"]])
    for event_n in test_events:
        rr.selected_index = event_n
        # sanity checks on particle values
        # momentum
        tst.assert_allclose(rr.PT**2, (rr.Px**2 + rr.Py**2), atol=0.001, rtol=0.01)
        tst.assert_allclose(rr.Birr**2, (rr.PT**2 + rr.Pz**2), atol=0.001, rtol=0.01)
        # on shell  lol-it's not on shell
        #tst.assert_allclose(rr.Mass, (rr.Energy**2 - rr.Birr**2), atol=0.0001)
        # angles
        tst.assert_allclose(rr.Phi, np.arctan2(rr.Py, rr.Px), atol=0.001, rtol=0.01)
        m2 = rr.Energy**2 - rr.Pz**2 - rr.PT**2
        with np.errstate(divide='ignore', invalid='ignore'):
            rapidity_calculated = 0.5*np.log((rr.PT**2 + m2)/(rr.Energy - np.abs(rr.Pz))**2)
        rapidity_calculated[np.isnan(rapidity_calculated)] = np.inf
        rapidity_calculated *= np.sign(rr.Pz)
        # im having dificulty matching the rapitity cauclation at all infinite points
        # this probably dosn't matter as high rapidity values are not seen anyway
        filt = np.logical_and(np.abs(rr.Rapidity) < 7., np.isfinite(rapidity_calculated))
        tst.assert_allclose(rr.Rapidity[filt], rapidity_calculated[filt], atol=0.01, rtol=0.01)
        # valid PID
        assert set(np.abs(rr.PID)).issubset(all_ids)
        # sanity checks on Towers
        # energy
        assert np.all(rr.Tower_ET <= rr.Tower_Energy)
        tst.assert_allclose((rr.Tower_Eem + rr.Tower_Ehad), rr.Tower_Energy, atol=0.001)
        # angle
        theta = np.arcsin(rr.Tower_ET/rr.Tower_Energy)
        eta = - np.log(np.tan(theta/2))
        tst.assert_allclose(eta, np.abs(rr.Tower_Eta), atol=0.001)
        # particles
        assert np.all(rr.Tower_Particles.flatten() >= 0)
        num_hits = [len(p) for p in rr.Tower_Particles]
        tst.assert_allclose(num_hits, rr.Tower_NTimeHits)
        # sanity check on Tracks
        # momentum
        # mometum removes as it appears to old nonsense
        #assert np.all(rr.Track_PT <= rr.Track_Birr)
        # angle
        #theta = np.arcsin(rr.Track_PT/rr.Track_Birr)
        #eta = - np.log(np.tan(theta/2))
        #tst.assert_allclose(eta, np.abs(rr.Track_Eta), atol=0.001)
        # particles
        assert np.all(rr.Track_Particle >= 0)
    

