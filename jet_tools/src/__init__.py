# this gets litrally everything
# which would be fine if there wern't some scripts mixed in there
# also, it would be best to rename src->core, 
# move some things out of core to other subpackages
# import everythign in core
# and import other subpackages as needed

#import os
#dir_path = os.path.split(__file__)[0]
#__all__ = [name.replace('.py', '') for name in os.listdir(dir_path)
#           if name.endswith('.py') and not name.startswith('_')]
#print(__all__)
#del os

# for now just manually specify stuff
# avoid DrawBarrel becuase it requires mayavi
__all__ = ['Components', 'FormVertices', 'JoinHepMCRoot', 'ReadHepmc', 'DrawTrees', 'CompareClusters', 'PDGNames', 'TrueTag', 'MergeHepmc', 'TreeWalker', 'FakeTracksTowers', 'RescaleJets', 'PlottingTools', 'ParameterInvestigation', 'TrueDenominator', 'ShapeVariables', 'MassPeaks', 'Constants', 'JetQuality', 'InputTools', 'AreaMeasures', 'SingleFormJets', 'OverlapPlotting', 'CompareDatasets', 'ParallelFormJets', 'FormJets', 'FormShower']
