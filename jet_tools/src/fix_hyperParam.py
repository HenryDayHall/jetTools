from jet_tools.src import Components
from jet_tools.src import CompareClusters
from ipdb import set_trace as st


def fix(variable):
    """
    

    Parameters
    ----------
    variable :
        

    Returns
    -------

    """
    options = {True: 'invarient', False: 'angular', 'Luclus': 'Luclus',
               'True': 'invarient', 'False': 'angular', 'Luclus': 'Luclus'}
    if variable in options:
        return options[variable]
    else:
        return variable


def fix_eventWise(eventWise, variable, fixing_function):
    """
    

    Parameters
    ----------
    eventWise :
        
    variable :
        
    fixing_function :
        

    Returns
    -------

    """
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    columns = [name for name in eventWise.columns if name.split('_', 1)[-1] == variable]
    hyperparam = [name for name in eventWise.hyperparameter_columns if name.split('_', 1)[-1] == variable]
    to_append = {}
    for name in columns:
        to_append[name] = fixing_function(getattr(eventWise, name))
    eventWise.append(**to_append)
    to_append = {}
    for name in hyperparam:
        to_append[name] = fixing_function(getattr(eventWise, name))
    eventWise.append_hyperparameters(**to_append)
    

#all_ew = Components.find_eventWise_in_dir("megaIgnore")
#while all_ew:
#    ew = all_ew.pop()
#    print(ew.save_name)
#    fix_eventWise(ew, "PhyDistance", fix)
#    del ew
#    

def fix_records(records, variable, fixing_function):
    """
    

    Parameters
    ----------
    records :
        
    variable :
        
    fixing_function :
        

    Returns
    -------

    """
    if isinstance(records, str):
        records = CompareClusters.Records(records)
    idx = records.indices[variable]
    n_rows = len(records.content)
    for row_n in range(n_rows):
        records.content[row_n][idx] = fixing_function(records.content[row_n][idx])
    records.write()

#fix_records("scans.csv", "PhyDistance", fix)

old_records = CompareClusters.Records("scans.csvbk")
old_ids = old_records.jet_ids
old_col = old_records.indices['PhyDistance']

new_records = CompareClusters.Records("scans.csv")
new_ids = new_records.jet_ids
new_col = new_records.indices['PhyDistance']

all_ew = Components.find_eventWise_in_dir("megaIgnore")
while all_ew:
    ew = all_ew.pop()
    print(ew.save_name)
    invarient_cols = [c for c in ew.hyperparameter_columns if c.endswith('PhyDistance')]
    if not invarient_cols:
        continue
    to_append = {}
    for col in invarient_cols:
        id_num = int(''.join([x for x in col.split('_', 1)[0] if x.isdigit()]))
        if id_num in old_records.jet_ids:
            old_value = old_records.content[old_ids.index(id_num)][old_col]
            new_value = fix(old_value)
            assert new_value is not None
            new_records.content[new_ids.index(id_num)][new_col] = new_value
            to_append[col] = new_value
    #ew.append_hyperparameters(**to_append)
    del ew
new_records.write()
