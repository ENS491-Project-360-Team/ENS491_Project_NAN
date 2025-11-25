# This file is a simple Python script which stores the chemical features in vector format. 
# This script requires a file (uniqueDrugs_cids.txt below) that contains PubChem CIDs of the drugs 
# and writes the results in another output file. 
# You can also manipulate this script according to your needs:
#   e.g. if you do not have PubChem CID of your drug of interest, 
#   you can use SMILES string of drug by making a small update on the Python script

import pychem as chem
from pychem import pychem
import numpy as np
from pychem.pychem import PyChem2d, PyChem3d

def get_chemical_data(id): # takes cid of drug as input
    d1 = PyChem2d()
    smi = d1.GetMolFromNCBI(id) 
    mol = d1.ReadMolFromSmile(smi)
    chemical_feat = {}
    chemical_feat.update(chem.constitution.GetConstitutional(mol))
    chemical_feat.update(chem.connectivity.GetConnectivity(mol))
    chemical_feat.update(chem.kappa.GetKappa(mol))
    chemical_feat.update(chem.bcut.CalculateBurdenVDW(mol))
    chemical_feat.update(chem.bcut.CalculateBurdenPolarizability(mol))
    chemical_feat.update(chem.estate.GetEstate(mol))
    chemical_feat.update(chem.basak.Getbasak(mol))
    chemical_feat.update(chem.geary.GetGearyAuto(mol))
    chemical_feat.update(chem.moran.GetMoranAuto(mol))
    chemical_feat.update(chem.moreaubroto.GetMoreauBrotoAuto(mol))
    chemical_feat.update(chem.molproperty.GetMolecularProperty(mol))
    chemical_feat.update(chem.moe.GetMOE(mol))

    return(chemical_feat)



drugs = np.loadtxt('uniqueDrugs_cids.txt',dtype='str', delimiter="\n") # read drug cid file, each line contains cid of a given drug
cid = str(drugs[0])
drug = get_chemical_data(cid)
col_names = drug.keys()
colCount = len(col_names)
chemicals = np.zeros((drugs.shape[0], colCount), dtype = np.float)
for k_id in range(colCount):
    chemicals[0,k_id] = drug[col_names[k_id]]

for i in range(1, drugs.shape[0]):
    drug = {}
    cid = str(drugs[i])
    dname = str(drugs[i])
    print(dname)
    drug = get_chemical_data(cid)
    for k_id in range(colCount):
        chemicals[i,k_id] = drug[col_names[k_id]]

np.savetxt('drugsDescriptors.csv', chemicals,delimiter=',') # save drug features