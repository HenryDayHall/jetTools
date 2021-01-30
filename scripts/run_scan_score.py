from jet_tools import ParallelFormJets, FormJets
import os, sys, time, shutil
jet_class_name = sys.argv[1].strip()
jet_class = getattr(FormJets, jet_class_name)
if jet_class_name in ["Traditional", "IterativeCone"]:
    scan_parameters = ParallelFormJets.scan_Traditional
    fix_parameters = ParallelFormJets.fix_Traditional
else:
    scan_parameters = ParallelFormJets.scan_spectral
    fix_parameters = ParallelFormJets.fix_spectral

# copy the eventWise
order = 'lo'
source = f"../megaIgnore/IRCchecks_noPTcut1/iridis_pp_to_jjj_{order}.awkd"
eventWise_path = f"../megaIgnore/IRCchecks_noPTcut1/iridis{order}_Scan_{jet_class_name}.awkd"
shutil.copyfile(source, eventWise_path)
end_time = time.time() + int(sys.argv[2])
ParallelFormJets.scan_score(eventWise_path, jet_class, end_time, scan_parameters, fix_parameters=fix_parameters, dijet_mass=40)
