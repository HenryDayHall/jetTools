""" File for testing the routeens in AlterLHE.py"""
from ipdb import set_trace as st
from tree_tagger import PDGNames, AlterLHE
from numpy import testing as tst
from tools import TempTestDir
import numpy as np
import ast
import os
pdg_ids = PDGNames.Identities()


def test_Particle():
    example = "       35  2    1    2    0    0 +1.4210854715e-14 +0.0000000000e+00 +7.6083744457e+01 5.0414878349e+02 4.9837461786e+02 0.0000e+00 0.0000e+00"
    example_values = (35, 2, 1, 2, 0, 0,  +1.4210854715e-14, +0.0000000000e+00,
                      +7.6083744457e+01, 5.0414878349e+02, 4.9837461786e+02, 0.0000e+00,
                      0.0000e+00)
    desired = {"MCPID": 35, "status": 2, "mother 1": 1, "mother 2": 2, "colour 1": 0,
               "colour 2": 0, "px": +1.4210854715e-14, "py": +0.0000000000e+00,
               "pz": +7.6083744457e+01, "e": 5.0414878349e+02, "m": 4.9837461786e+02,
               "propper lifetime": 0.0000e+00, "spin": 0.0000e+00}
    particle = AlterLHE.Particle(example)
    for key in desired:
        tst.assert_allclose(desired[key], particle[key], err_msg=f"{key} not matching")
    # to test if the string is close, just check each number
    for str_part_out, example_value in zip(str(particle).split(), example_values):
        value_out = ast.literal_eval(str_part_out)
        assert type(value_out) == type(example_value)
        tst.assert_allclose(value_out, example_value)
    
# Event ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_Event():
    # def test___init__():
    # def test___str__():
    example = \
""" 9      1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01
       35  2    1    2    0    0 +1.4210854715e-14 +0.0000000000e+00 +7.6083744457e+01 5.0414878349e+02 4.9837461786e+02 0.0000e+00 0.0000e+00
        5  1    4    4  502    0 -7.4971530554e+01 +1.6425317942e+02 +1.1211674416e+02 2.1258431660e+02 4.7000000000e+00 0.0000e+00 1.0000e+00
"""
    event = AlterLHE.Event(example)
    assert example.startswith(event.event_info)
    assert len(event.particles) == 2
    found_str = str(event)
    assert found_str.split(os.linesep, 1)[0] == example.split(os.linesep, 1)[0]
    # cannot garentee other lines are exact, but there shoudl be 3 lines
    assert len(found_str.split(os.linesep)) == 3
    # def test_increment_particle_count():
    event.increment_particle_count(3)
    new_info = " 12 1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01"
    assert event.event_info == new_info
    # def test_add_split():  start with a soft split
    # reset the event
    event = AlterLHE.Event(example)
    event.add_split(split_type='soft', max_split=0)
    new_info = " 11 1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01"
    assert event.event_info == new_info
    assert len(event.particles) == 4
    assert np.sum([p["MCPID"] == 5 for p in event.particles]) == 2
    decayed_particle = next(p for p in event.particles if p["MCPID"] == 5 and p["status"] == 2)
    after_decay = next(p for p in event.particles if p["MCPID"] == 5 and p["status"] == 1)
    coords = ["e", "px", "py", "pz"]
    tst.assert_allclose(*[[particle[coord] for coord in coords]
                          for particle in [after_decay, decayed_particle]])
    soft_radiation = next(p for p in event.particles if p["MCPID"] in 
                          AlterLHE.Event.emissions and p["status"] == 1)
    tst.assert_allclose([soft_radiation[coord] for coord in coords], np.zeros(len(coords)))
    # now try a collinear split
    event = AlterLHE.Event(example)
    event.add_split(split_type='collinear', max_split=0)
    new_info = " 11 1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01"
    assert event.event_info == new_info
    assert len(event.particles) == 4
    assert np.sum([p["MCPID"] == 5 for p in event.particles]) == 2
    decayed_particle = next(p for p in event.particles if p["MCPID"] == 5 and p["status"] == 2)
    after_decay = next(p for p in event.particles if p["MCPID"] == 5 and p["status"] == 1)
    colinear_split = next(p for p in event.particles if p["MCPID"] in 
                          AlterLHE.Event.emissions and p["status"] == 1)
    coords = ["px", "py", "pz"]
    tst.assert_allclose(*[[particle[coord] for coord in coords]
                          for particle in [after_decay, colinear_split]])
    tst.assert_allclose([colinear_split[coord] for coord in coords],
                        np.array([decayed_particle[coord] for coord in coords])*0.5)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_collinear_kinematics():
    input_momentum = np.zeros(3)
    particle_1, particle_2 = AlterLHE.collinear_kinematics(input_momentum, 0)
    tst.assert_allclose(particle_1, 0)
    tst.assert_allclose(particle_2, 0)
    max_val = 5
    particle_1, particle_2 = AlterLHE.collinear_kinematics(input_momentum, max_val)
    assert np.all(np.abs(particle_1) < max_val)
    assert np.all(np.abs(particle_2) < max_val)
    tst.assert_allclose(particle_1 + particle_2, 0)
    # non zero input
    input_momentum = np.arange(3, dtype=float)
    particle_1, particle_2 = AlterLHE.collinear_kinematics(input_momentum, 0)
    tst.assert_allclose(particle_1, input_momentum*0.5)
    tst.assert_allclose(particle_2, input_momentum*0.5)
    particle_1, particle_2 = AlterLHE.collinear_kinematics(input_momentum, max_val)
    tst.assert_allclose(particle_1 + particle_2, input_momentum)
    


def test_soft_kinematics():
    input_momentum = np.zeros(3)
    particle_1, particle_2 = AlterLHE.soft_kinematics(input_momentum, 0)
    tst.assert_allclose(particle_1, 0)
    tst.assert_allclose(particle_2, 0)
    max_val = 5
    particle_1, particle_2 = AlterLHE.soft_kinematics(input_momentum, max_val)
    assert np.all(np.abs(particle_1) < max_val)
    assert np.all(np.abs(particle_2) < max_val)
    tst.assert_allclose(particle_1 + particle_2, 0)
    # non zero input
    input_momentum = np.arange(3, dtype=float)
    particle_1, particle_2 = AlterLHE.soft_kinematics(input_momentum, 0)
    tst.assert_allclose(particle_1, input_momentum)
    tst.assert_allclose(particle_2, 0)
    particle_1, particle_2 = AlterLHE.soft_kinematics(input_momentum, max_val)
    tst.assert_allclose(particle_1 + particle_2, input_momentum)



def test_apply_to_events():
    with TempTestDir("temp") as dir_name:
        # need an example
        example_file = \
"""
<LesHouchesEvents version="3.0">
<header>
<MGVersion>
2.6.3.2
</MGVersion>
</header>
<event>
 9      1 +4.5052000e+02 4.98374600e+02 7.81860800e-03 1.01420400e-01
       -1 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +2.9011626398e+02 2.9011626398e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        5  1    4    4  502    0 -7.4971530554e+01 +1.6425317942e+02 +1.1211674416e+02 2.1258431660e+02 4.7000000000e+00 0.0000e+00 1.0000e+00
       -5  1    4    4    0  502 -4.6386869919e+01 -1.3849390228e+01 +1.9818207888e+01 5.2520459580e+01 4.7000000000e+00 0.0000e+00 1.0000e+00
<mgrwt>
<rscale>  0 0.49837462E+03</rscale>
<asrwt>0</asrwt>
<pdfrwt beam="1">  1        1 0.32928080E-01 0.49837462E+03</pdfrwt>
<pdfrwt beam="2">  1       -1 0.44633271E-01 0.49837462E+03</pdfrwt>
<totfact> 0.68232681E+02</totfact>
</mgrwt>
<rwgt>
<wgt id='1'> +4.5105783e+02 </wgt>
<wgt id='2'> +4.5092593e+02 </wgt>
</rwgt>
</event>
</LesHouchesEvents>
"""
        input_name = os.path.join(dir_name, "inp.lhe")
        output_name = os.path.join(dir_name, "out.lhe")
        with open(input_name, 'w') as input_file:
            input_file.write(example_file)
        AlterLHE.apply_to_events(input_name, output_name, 'soft')
        assert os.path.exists(output_name)
        with open(input_name, 'r') as input_file:
            input_text = input_file.read()
        assert input_text == example_file
        with open(output_name, 'r') as output_file:
            output_text = output_file.read()
        assert len(input_text.split(os.linesep)) + 2 == len(output_text.split(os.linesep))

