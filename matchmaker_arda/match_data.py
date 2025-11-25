import pandas as pd
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from rdkit import DataStructs
import os



os.chdir(r"C:\Users\ardat\OneDrive - Sabancı Üniversitesi\Desktop\bitirme\matchmaker")

# 1) Load your GDSC pancreas file
df = pd.read_csv("gdsc_pancreas_flexiscatter.csv")

# Expect columns: Tissue, Cancer Type, Combination, Cell line name, SIDM ID, Bliss Emax, Delta IC50

# 2) Split "Combination" into drug1 and drug2
df[["drug1", "drug2"]] = df["Combination"].str.split(" \+ ", expand=True)

# 3) Build comb_data.tsv in a MatchMaker-like format
# MatchMaker expects a column named synergy_loewe, but here it's actually Bliss.
# We'll map Bliss Emax into a column called synergy_loewe for code compatibility.
comb_data = df[["drug1", "drug2", "Cell line name", "Bliss Emax"]].rename(
    columns={
        "Cell line name": "cell_line",
        "Bliss Emax": "synergy_loewe"  # NOTE: this is actually Bliss, not Loewe
    }
)

comb_data.to_csv("comb_data.tsv", sep="\t", index=False)
print("Saved comb_data.tsv with shape:", comb_data.shape)

# 4) Extract unique drug list
all_drugs = pd.unique(pd.concat([comb_data["drug1"], comb_data["drug2"]], ignore_index=True))
print(f"Found {len(all_drugs)} unique drugs.")

# 5) Get SMILES from PubChem
def get_smiles_from_name(name):
    try:
        res = pcp.get_compounds(name, "name")
        if len(res) == 0:
            return None
        return res[0].canonical_smiles
    except Exception as e:
        print("Error retrieving SMILES for", name, ":", e)
        return None

drug_smiles_records = []
for d in all_drugs:
    smiles = get_smiles_from_name(d)
    print(d, "→", smiles)
    drug_smiles_records.append({"drug": d, "smiles": smiles})
    time.sleep(0.2)  # be gentle with PubChem

drug_smiles_df = pd.DataFrame(drug_smiles_records)
drug_smiles_df.to_csv("drug_smiles.csv", index=False)
print("Saved drug_smiles.csv. Check for any missing SMILES (NaN).")

def morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)  # <-- use DataStructs here
    return arr


rows = []
missing_smiles = []

for _, row in drug_smiles_df.iterrows():
    drug = row["drug"]
    smiles = row["smiles"]
    if pd.isna(smiles):
        print("No SMILES for", drug, "- please fix manually in drug_smiles.csv if needed.")
        missing_smiles.append(drug)
        continue
    fp = morgan_fp(smiles)
    if fp is None:
        print("RDKit failed to parse SMILES for", drug)
        missing_smiles.append(drug)
        continue
    rows.append([drug] + fp.tolist())

if missing_smiles:
    print("WARNING: missing/failed fingerprints for these drugs:", missing_smiles)

if not rows:
    raise RuntimeError("No fingerprints generated – check SMILES and retry.")

fp_cols = ["drug"] + [f"fp_{i}" for i in range(len(rows[0]) - 1)]
drug_features_df = pd.DataFrame(rows, columns=fp_cols)
drug_features_df.to_csv("drug_features_morgan.csv", index=False)
print("Saved drug_features_morgan.csv with shape:", drug_features_df.shape)

# 7) Build drug1_chemicals.csv and drug2_chemicals.csv aligned to comb_data.tsv

feature_cols = [c for c in drug_features_df.columns if c != "drug"]
feat_dict = {
    row["drug"]: row[feature_cols].values
    for _, row in drug_features_df.iterrows()
}

drug1_mat = []
drug2_mat = []
bad_rows = 0

for _, row in comb_data.iterrows():
    d1 = row["drug1"]
    d2 = row["drug2"]
    if d1 not in feat_dict or d2 not in feat_dict:
        bad_rows += 1
        # If you don't want to drop rows, you could also fill with zeros.
        continue
    drug1_mat.append(feat_dict[d1])
    drug2_mat.append(feat_dict[d2])

if bad_rows > 0:
    print(f"WARNING: {bad_rows} rows in comb_data.tsv had missing drug features and were skipped.")
    # If you skip rows, you should also save a filtered comb_data:
    valid_indices = [i for i, row in enumerate(comb_data.itertuples(index=False)) 
                     if getattr(row, 'drug1') in feat_dict and getattr(row, 'drug2') in feat_dict]
    comb_data_filtered = comb_data.iloc[valid_indices].reset_index(drop=True)
    comb_data_filtered.to_csv("comb_data_filtered.tsv", sep="\t", index=False)
    print("Saved comb_data_filtered.tsv aligned with chemical feature matrices.")
else:
    comb_data_filtered = comb_data

drug1_mat = np.vstack(drug1_mat)
drug2_mat = np.vstack(drug2_mat)

pd.DataFrame(drug1_mat).to_csv("drug1_chemicals.csv", index=False, header=False)
pd.DataFrame(drug2_mat).to_csv("drug2_chemicals.csv", index=False, header=False)

print("Saved drug1_chemicals.csv with shape:", drug1_mat.shape)
print("Saved drug2_chemicals.csv with shape:", drug2_mat.shape)
print("Done.")
