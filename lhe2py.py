import numpy as np 
import os
import pandas as pd 
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import csv

pd.set_option('display.max_rows', 100) #display more rows when printing dataframe

#takes a .lhe file, and cleans and converts into a form that can be read into a pandas dataframe, saves output as new .csv file
def lhe_cleaner(file_name): #don't include .lhe part!
    suffix = '.lhe'
    if not file_name.endswith(suffix):
        file_name = file_name+suffix
    clean_name = file_name[:-4] + '_clean.csv'
    delimiter = ','
    select_tag = 'event'
    start_tag = '<' + select_tag + '>'
    stop_tag = '</' + select_tag + '>'
    n_columns = 13
    event_divider = ['0' for _ in range(n_columns)]
    column_names = ['PDG','Status','MOTH1','MOTH2','ICOL1',
                    'ICOL2','Px','Py','Pz','E','M','c_tau','Spin']
    # always use a context manager for files
    with open(file_name) as lhe_file, open(clean_name,'w') as clean_file:
        writer = csv.writer(clean_file, delimiter=delimiter)
        INSIDE = 0; HEADER = 1; OUTSIDE = 2
        position = OUTSIDE  # toggle to keep track if we are in the tag block of intrest
        for line in lhe_file:
            # no matter what a stop tag means exiting  block
            if stop_tag in line:
                position = OUTSIDE
                writer.writerow(event_divider)
            elif start_tag in line: # first line in the block is the header
                position = HEADER
            elif position == HEADER: # then next line is inside
                position = INSIDE
            elif position == INSIDE and '<' not in line:
                # this is the only bit we keep, and only if its got no tags in
                writer.writerow(line.split())
    df = pd.read_csv(clean_name, sep=delimiter, header=None, names=column_names) #create dataframe
    # add some rows
    df["Eta"] = np.arctanh(df["Pz"]/np.sqrt(df["Px"]**2+df["Py"]**2+df["Pz"]**2)) #compute eta's of particles
    df['event_count'] = (df['PDG'] == 0).cumsum()+1 #add event counter
    df = df.drop(df[df.PDG ==0].index)
    df.to_csv(clean_name, index=None, sep=delimiter) #rewrite csv file with NaNs removed
    return df

#creates new csv file containing only the b jets from each event 
def b_jets(file_name): 
    suffix = '_bjets.csv'
    input_suffix = '_clean.csv'
    # deal with various possible input file names
    if file_name.endswith(input_suffix):
        file_name = file_name[:-len(input_suffix)]
    elif file_name.endswith(suffix):
        file_name = file_name[:-len(suffix)]
    elif file_name.endswith('.lhe'):
        file_name = file_name[:-4]
    # get the input
    if os.path.exists(file_name + input_suffix):
        # don't repeat the cleaning if the fiel exisits
        df = pd.read_csv(file_name + input_suffix)
    else:
        df = lhe_cleaner(file_name)
    file_name = file_name+suffix
    df = df.drop(columns=["MOTH1","MOTH2","ICOL1","ICOL2","M","c_tau","Spin"]) #only interested in a few of the values
    df = df.drop(df[(df.PDG != 5) & (df.PDG != -5) & (df.PDG != 0)].index) #keep only b quarks and jets (PDG +/-5 and Status_Code 3/1) (and event separation 0's)
    df = df.drop(df[df.Status == 3].index) #remove b quarks, only interested in jets
    df = df.reset_index(drop=True)
    df.to_csv(file_name, index=None) #output text file containing only bjets, 46x smaller than original lhe!
    return df

#b jet analysis, creates 4 dataframes containing the (leading, sub leading, sub^2 leading and sub^3 leading jet) b jets in each event
def eta_E(file_name):
    suffix = '_bjets.csv'
    # deal with various possible input file names
    if not file_name.endswith(suffix):
        if file_name.endswith('_clean.csv'):
            file_name = file_name[:-10]
        if file_name.endswith('.lhe'):
            file_name = file_name[:-4]
        file_name = file_name+suffix
    df = pd.read_csv(file_name)
    ###Kinematic cut below
    df = df.drop(df[(df.Eta > 2.5) | (df.Eta < -2.5)].index) #eta cut, as it is not applied on the .lhe file 
    #create new dataframe for each energy ordered jet
    # need to assign colum names now so that empty data frames have correct column names
    bjets_dataframes = [pd.DataFrame(columns=df.columns) for _ in range(4)]
    events = df.groupby('event_count') #group the bjets into their events
    for count, group in events:
        group = group.sort_values('E',ascending=False) #sort b jets in decreasing energy
        for jet_n in range(min(len(group), 4)):
            bjets_dataframes[jet_n] = bjets_dataframes[jet_n].append(group.iloc[jet_n])
    return bjets_dataframes #ouput is list containing the 4 dataframes, one for each energy ranked bjet across all events

def eta_E_plot(file_name):
    data = eta_E(file_name)
    eta1 = data[0].Eta
    E1 = data[0].E
    eta2 = data[1].Eta
    E2 = data[1].E
    eta3 = data[2].Eta
    E3 = data[2].E
    eta4 = data[3].Eta
    E4 = data[3].E #pick out relavent dataframe from eta_E output
    ax1 = plt.subplot(221)
    ax1.set_title('Jet 1')
    plt.plot(eta1,E1 , 'ro')
    ax1.set_xlabel('eta')
    ax1.set_ylabel('E')
    ax2 = plt.subplot(222)
    ax2.set_title('Jet 2')
    plt.plot(eta2,E2 , 'ro')
    ax2.set_xlabel('eta')
    ax2.set_ylabel('E')
    ax3 = plt.subplot(223)
    ax3.set_title('Jet 3')
    plt.plot(eta3,E3 , 'ro')
    ax3.set_xlabel('eta')
    ax3.set_ylabel('E')
    ax4 = plt.subplot(224)
    ax4.set_title('Jet 4')
    plt.plot(eta4,E4 , 'ro')
    ax4.set_xlabel('eta')
    ax4.set_ylabel('E') #standard plotting routine
    plt.show()

###SET FILE HERE!
#file = 'pure_phys'
#eta_E_plot(file)
#print(eta_E(file)[0])
#b_jets(file) #only need to do once!


