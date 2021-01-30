from jet_tools import ParallalFormJets, FormJets
import os, sys, time, shutil
jet_class_name = sys.argv[1].strip()
jet_class = getattr(FormJets, jet_class_name)
if jet_class_name in ["Traditional", "IterativeCone"]:
    scan_parameters = ParallalFormJets.scan_Traditional
    fix_parameters = ParallalFormJets.fix_Traditional
else:
    scan_parameters = ParallalFormJets.scan_spectral
    fix_parameters = ParallalFormJets.fix_spectral

# copy the eventWise
order = 'lo'
source = f"../megaIgnore/IRCchecks_noPTcut1/iridis_pp_to_jjj_{order}.awkd"
eventWise_path = f"../megaIgnore/IRCchecks_noPTcut1/iridis{order}_Scan_{jet_class_name}.awkd"
shutil.copyfile(source, eventWise_path)
end_time = time.time() + int(sys.argv[2])
ParallalFormJets.scan_score(eventWise_path, jet_class, end_time, scan_parameters, fix_parameters=fix_parameters, dijet_mass=40)
