import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100) #display more rows when printing dataframe

#takes a .lhe file, and cleans and converts into a form that can be read into a pandas dataframe, saves output as new .txt file
def lhe_cleaner(file_name): #don't include .lhe part!
	raw = open(file_name+".lhe").read().splitlines() #open file
	raw = raw[115:] #delete header stuff
	lhe = []
	for i in range(len(raw)):
		raw[i] = raw[i].strip() #strip whitespace  
		raw[i] = re.sub(' +', ' ',raw[i]) #remove duplicate spacing
		if raw[i] == '</event>':
			raw[i] = "0 0 0 0 0 0 0 0 0 0 0 0 0" #split events with a line of zeroes
		if raw[i][0] != '<':
			lhe.append(raw[i]) #fill new lhe object with the data, removing <event> flags
	open(file_name+'_clean.txt','w').write('\n'.join(lhe)) #writes data to new 'clean' text file
	df = pd.read_csv(file_name+'_clean.txt', sep=" ", header=None, names=['PDG','Status','MOTH1','MOTH2','ICOL1','ICOL2','Px','Py','Pz','E','M','c_tau','Spin']) #create dataframe
	df = df.dropna() #removes event information header
	df = df.reset_index(drop=True)
	df["Eta"] = np.arctanh(df["Pz"]/np.sqrt(df["Px"]**2+df["Py"]**2+df["Pz"]**2)) #compute eta's of particles
	df['event_count'] = (df['PDG'] == 0).cumsum()+1 #add event counter
	df = df.drop(df[df.PDG ==0].index)
	df.to_csv(file_name+'_clean.txt', index=None) #rewrite txt file with NaNs removed
	return df

#creates new txt file containing only the b jets from each event 
def b_jets(file_name): 
	df = lhe_cleaner(file_name)
	df = df.drop(columns=["MOTH1","MOTH2","ICOL1","ICOL2","M","c_tau","Spin"]) #only interested in a few of the values
	df = df.drop(df[(df.PDG != 5) & (df.PDG != -5) & (df.PDG != 0)].index) #keep only b quarks and jets (PDG +/-5 and Status_Code 3/1) (and event separation 0's)
	df = df.drop(df[df.Status == 3].index) #remove b quarks, only interested in jets
	df = df.reset_index(drop=True)
	df.to_csv(file_name+'_bjets.txt', index=None) #output text file containing only bjets, 46x smaller than original lhe!
	return df

#b jet analysis, creates 4 dataframes containing the (leading, sub leading, sub^2 leading and sub^3 leading jet) b jets in each event
def eta_E(file_name):
	df = pd.read_csv(file_name+'_bjets.txt')
	###Kinematic cut below
	df = df.drop(df[(df.Eta > 2.5) | (df.Eta < -2.5)].index) #eta cut, as it is not applied on the .lhe file 
	bjets1 = pd.DataFrame() #create new dataframe for each energy ordered jet
	bjets2 = pd.DataFrame()
	bjets3 = pd.DataFrame()
	bjets4 = pd.DataFrame()
	events = df.groupby('event_count') #group the bjets into their events
	for count, group in events:
		group = group.sort_values('E',ascending=False) #sort b jets in decreasing energy
		if len(group) == 4: #each event contains n jets, if n < 4 we only fill the top n (out of 4) dataframes
			bjets1 = bjets1.append(group.iloc[0])#take desired b jet (leading, sub leading, sub sub leading...)
			bjets2 = bjets2.append(group.iloc[1])
			bjets3 = bjets3.append(group.iloc[2])
			bjets4 = bjets4.append(group.iloc[3])
		if len(group) == 3: #like wise if event contains only 3/2/1 bjets
			bjets1 = bjets1.append(group.iloc[0])
			bjets2 = bjets2.append(group.iloc[1])
			bjets3 = bjets3.append(group.iloc[2])
		if len(group) == 2:
			bjets1 = bjets1.append(group.iloc[0])
			bjets2 = bjets2.append(group.iloc[1])
		if len(group) == 1:
			bjets1 = bjets1.append(group.iloc[0])
		if len(group) == 0:
			continue #if event contains no bjets, we skip
	return bjets1, bjets2, bjets3, bjets4 #ouput is tuple containing the 4 dataframes, one for each energy ranked bjet across all events

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
file = 'pure_phys'
#eta_E_plot(file)
print eta_E(file)[0]
#b_jets(file) #only need to do once!


