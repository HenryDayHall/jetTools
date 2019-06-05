# coding: utf-8
import ReadSQL
event, tracks, towers = ReadSQL.main()
import FormShower
showers = FormShower.get_showers(event)
import Components
obs = Components.Observables(event, towers=towers, tracks=tracks)
import FormJets
psudojets = FormJets.PsudoJets(obs)
psudojets.assign_mothers()
jets = psudojets.split()
import DrawTrees
graph = DrawTrees.DotGraph(showers[-1], observables=obs, jet=jets[1])
