#!/usr/bin/env python
# coding: utf-8
# Import some stuff
import pandas as pd
import math
from collections import Counter
from collections import defaultdict
import itertools
import argparse
import sys
import ast
import os
import numpy as np
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from IPython.display import SVG, display
import re
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
def dedupe_bonds(data_dict):
    cleaned = {}
    for parent_smiles, bonds in data_dict.items():
        seen = set()         # will hold frozensets of SMILES pairs
        new_bonds = {}
        # iterate in original order—this will keep the first occurrence
        for bond_idx, info in bonds.items():
            pair = info.get('SMILES', [])
            key = frozenset(pair)
            if key not in seen:
                seen.add(key)
                new_bonds[bond_idx] = info
            # else:  duplicate, so skip it
        cleaned[parent_smiles] = new_bonds
    return cleaned
def canonicalize_smiles(smiles): # Cannonicalize the smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        log_mess(f" Could not find mol {smiles}.")
        return None
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        mol=Chem.AddHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        #print('No SMILES')
        return None
def remove_dummy_atoms_and_add_radicals(mol):
    mol = Chem.RWMol(mol)  # Convert to editable molecule
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]  # Find dummy atoms
    for dummy_idx in sorted(dummy_atoms, reverse=True):  
        bonds = mol.GetAtomWithIdx(dummy_idx).GetBonds()
        if bonds:  # Ensure the dummy atom was bonded
            neighbor = bonds[0].GetOtherAtomIdx(dummy_idx)  # Get bonded atom
            atom = mol.GetAtomWithIdx(neighbor)
            # Set radical on the bonded atom
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
        mol.RemoveAtom(dummy_idx)  # Remove dummy atom
    return mol
def log_mess(message, log_file="log.txt"):
    with open(log_file, "a") as f:
        f.write(message + "\n")
def search_smiles_in_dataframe(fragment_smiles_list, df,duplicates,final_fragment_list,Not_found):
    # duplicates sometimes got made as naming conventions and workflows changed, this ensure all duplicates are the ones that are already known and removed.
    # any other duplicates will be considered invalid for both
    # Efficient vectorized search using 'isin' for checking multiple fragments
    matching_rows = df['Canonical_SMILES'][df['Canonical_SMILES'].isin(fragment_smiles_list)]
    Molecule_name=''

    # Print out which fragments couldn't be found
    not_found = [frag for frag in fragment_smiles_list if frag not in matching_rows.values]

    if not_found:
        log_mess(f"The following fragments couldn't be found in the dataframe:")
        print(f"The following fragments couldn't be found in the dataframe:")
        for frag in not_found:
            Not_found.append(frag)
            log_mess(f": {frag}")
            

    for a in matching_rows:
        if not duplicates.empty and 'Canonical_SMILES' in duplicates.columns:
            if a in duplicates['Canonical_SMILES'].values:  # Check if the SMILES is in the duplicates list
                log_mess('Found duplicate',a)
                
            else: # grab the values
                Molecule_find=df[df['Canonical_SMILES'] == a]['Canonical_SMILES'].iloc[0]
                #print('Grabbing Canonical_Smiles',Molecule_find)

                Molecule_name=Molecule_find
                if Molecule_find not in final_fragment_list:
                    final_fragment_list.append(Molecule_find)
        else: # grab the values
            Molecule_find=df[df['Canonical_SMILES'] == a]['Canonical_SMILES'].iloc[0]
            #print('Grabbing Canonical_Smiles',Molecule_find)

            Molecule_name=Molecule_find
            if Molecule_find not in final_fragment_list:
                final_fragment_list.append(Molecule_find)
            #method_value_dict                                     
    # Return the matching rows
    return final_fragment_list, Molecule_name, Not_found
def add_fragments(frag_names,molecule_data):
    #print('Attempting these fragments:',frag_names)
    fragment_methods={}
    fragment_enthalpy_sums={}
    if '' in frag_names:
        return None
    else:
        for frag_name in frag_names:
            frag_data = molecule_data.get(frag_name)
            if frag_data:
                methods = set(method for method, _, _ in frag_data)  # Get a set of methods for this fragment
                fragment_methods[frag_name] = methods
            else:
                log_mess(f"Fragment '{frag_name}' not found in molecule data.")
                return None
        common_methods = set.intersection(*fragment_methods.values())  # Find methods common to both fragments
        for frag_name in frag_names:
            frag_data = molecule_data.get(frag_name)
            #print("This is the frag_name:",frag_name)
            if frag_data:
                for method, method_description, enthalpy in frag_data:
                    # Only add enthalpy for common methods
                    if method in common_methods:
                        if method not in fragment_enthalpy_sums:
                            #print(method)
                            #print(enthalpy)
                            fragment_enthalpy_sums[method] = enthalpy
                        else:
                            #print(method)
                            #print(enthalpy)
                            fragment_enthalpy_sums[method] += enthalpy

            else:
                log_mess(f"Fragment '{frag_name}' not found in molecule data.")
                return None

    return fragment_enthalpy_sums
def calculate_enthalpy(Parent_PFAS,Enthalpy_Data):
    full_molecule_enthalpy = {}  
    for method, value in Parent_PFAS.items():
        if method in Enthalpy_Data.keys():

            full_molecule_enthalpy[method]=(Enthalpy_Data[method]-value)*627.5

    return full_molecule_enthalpy                   

def enthalpy_parent(Parent_PFAS,molecule_data):
    #print('Attempting this:',Parent_PFAS)
    Parent_PFAS_methods={}
    Parent_PFAS_enthalpy={}
    if Parent_PFAS == None:
        return None
    else:

        parent_data = molecule_data.get(Parent_PFAS)
        if parent_data:
            for method, method_description, enthalpy in parent_data:

                Parent_PFAS_enthalpy[method] = enthalpy

        else:
            log_mess(f"Parent_PFAS '{Parent_PFAS}' not found in molecule data.")
            return None  
        return Parent_PFAS_enthalpy

def has_large_imag(freq_str):
    if not isinstance(freq_str, str):
        return False  # skip non-string (e.g., False, NaN)
    try:
        freqs = ast.literal_eval(freq_str)  # string -> list
        if not isinstance(freqs, (list, tuple)):
            return False
        return any(abs(f) > 100 for f in freqs)
    except (ValueError, SyntaxError):
        return False  # skip invalid strings
    
def step_1_parent(PFAS_list_smiles):
    PFAS_list=[]
    for PFAS in PFAS_list_smiles:
        PFAS_list.append(canonicalize_smiles(PFAS))

    Parent_PFAS = {}
    Not_found_list=[]

    for parent_smiles in PFAS_list:
        #print(parent_smiles)
        parent_mol = Chem.MolFromSmiles(parent_smiles)

        if parent_mol is None:
            raise ValueError("Error parsing SMILES!")

        # Find all single bond indices
        parent_mol = Chem.AddHs(parent_mol)
        single_bond_indices = [
            bond.GetIdx() for bond in parent_mol.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE
        ]
        #print(single_bond_indices)

        # Annotate bond indices
        for bond in parent_mol.GetBonds():
            bond_idx = bond.GetIdx()

            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            # if bond_idx in single_bond_indices:
            #     print(f"Bond IDX {bond_idx}: Atoms {atom1},{atom2}")

        # Compute 2D coordinates
        Chem.rdDepictor.Compute2DCoords(parent_mol)
        for atom in parent_mol.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        # Store with bond indices as keys to access specific BDE fragments
        for atom in parent_mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber") 


        Parent_PFAS[parent_smiles] = {}
        fragment_mols = []
        for bond_idx in single_bond_indices:
            # Process fragments as before
            fragmented_mol = Chem.FragmentOnBonds(parent_mol, [bond_idx], addDummies=True)
            fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)

            final_fragments = []
            fragment_smiles_list = []  # Store cleaned SMILES

            for frag in fragments:
                clean_mol = remove_dummy_atoms_and_add_radicals(frag)

                # Ensure no atom indices
                for atom in clean_mol.GetAtoms():
                    atom.ClearProp("molAtomMapNumber") 

                smiles_mol = Chem.MolToSmiles(clean_mol, canonical=True)
                final_fragments.append(clean_mol)
                fragment_smiles_list.append(smiles_mol)

            # Store by bond index
            Parent_PFAS[parent_smiles][bond_idx] = {
                "molecules": final_fragments,  # List of RDKit Mol objects
                "SMILES": fragment_smiles_list  # List of corresponding SMILES
            }
    return Parent_PFAS,PFAS_list

def step_2(Parent_PFAS,df_full,df,duplicates,PFAS_list):
    Not_found_list=[]
    Parent_PFAS_no_duplicates=dedupe_bonds(Parent_PFAS)
    for molecule_smiles in Parent_PFAS_no_duplicates:
        #print('IN this')
        #print('Is this the cannonical smiles?:',[molecule_smiles])
        final_fragment_list=[]# list of all non duplicate fragments to grab data for
        final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([molecule_smiles],df,duplicates,final_fragment_list,Not_found_list)

        Parent_PFAS[molecule_smiles]['Molecule_name']=name    
        for bond_idx, data in Parent_PFAS_no_duplicates[molecule_smiles].items():
            #print(f"Bond Index: {bond_idx}")
            #print("Fragment SMILES:", data["SMILES"][0])
            if bond_idx != 'Molecule_name':
                data['Molecule_name']=[] 
                final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([data["SMILES"][0]],df,duplicates,final_fragment_list,Not_found_list)  
                data['Molecule_name'].append(name)
                final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([data["SMILES"][1]],df,duplicates,final_fragment_list,Not_found_list)    
                data['Molecule_name'].append(name)
    # Now collect the data
        #print('This is the final_fragment_list:',final_fragment_list)
        filtered_df = df_full[df_full['Canonical_SMILES'].isin(final_fragment_list)]
        pd.set_option('display.width', None)   # auto‑detect width
        pd.set_option('display.max_columns', None)
        #print(filtered_df[filtered_df['Method']==35])
        # Initialize an empty dictionary to store methods and Enthalpy for each molecule
        molecule_data = {}
        # Iterate through the filtered DataFrame
        for _, row in filtered_df.iterrows():
            molecule = row['Canonical_SMILES']
            #print(molecule)
            # if row['Canonical_SMILES'] == '[H][C]([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F' or row['Canonical_SMILES']=='[H]O[S](=O)=O' or row['Canonical_SMILES']=='[H]OS(=O)(=O)C([H])([H])C([H])([H])[C](F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':
            #     print(row['Molecule'])
            #     print(row['Enthalpy'])
            #     print(row['Method_Name'])
            method = row['Method_Name']
            method_category = row['Method_Category']
            enthalpy = row['Enthalpy']
            #print(enthalpy)

            # Only store data for molecules in final_fragment_list
            if molecule in final_fragment_list and pd.notna(enthalpy) and pd.notna(method_category):

                # If the molecule is already in the dictionary, append the method and enthalpy
                if molecule not in molecule_data:
                    molecule_data[molecule] = []

                # Append the method and enthalpy as a tuple
                molecule_data[molecule].append((method, method_category,enthalpy))

        # Now that you have all the data do the fragments
        #print(Parent_PFAS[molecule_smiles]['Molecule_name'])
        Enthalpy_parent=enthalpy_parent(Parent_PFAS[molecule_smiles]['Molecule_name'],molecule_data)
        for bond_idx, data in Parent_PFAS_no_duplicates[molecule_smiles].items():

            if bond_idx != 'Molecule_name':
                data['BDE']={}
                Enthalpy_Data={}
                Enthalpy_Data=add_fragments(data['Molecule_name'],molecule_data)

                if Enthalpy_Data and Enthalpy_parent:
                    Final_Enthalpy=calculate_enthalpy(Enthalpy_parent,Enthalpy_Data)
                    data['BDE']=Final_Enthalpy
                    #print(data)
    list_of_head_groups=['(=O)=O','OS(=O)(=O)','OC(=O)','[O]C(=O)','O=[C]','O[S](=O)(=O)','O[C](=O)','[H]O[C]=O',
                        '[O-]S(=O)(=O)','[O-]C(=O)','O=[C]','O=S(=O)([O-])','C(=O)[O-]','O=[S](=O)[O-]','S(=O)(=O)[O-]','O=C([O-])']
    exact_head_groups=['[H][C](F)F','(F)[C](F)F','[H][C]([H])F','[H][C]([H])[H]','F[C](F)F']
    problem_smiles=['FC(F)(F)F','[H]C(F)(F)F','[H]C([H])(F)F','[H]C([H])([H])F','[H]C([H])([H])[H]']
    head_charge='IP_Enthalpy' # +1 no electrons
    tail_charge='EA_Enthalpy' # -1 both electrons
    tail_groups=[]
    head_groups=[]
    for molecule_smiles in Parent_PFAS_no_duplicates:
        #print(molecule_smiles)
        final_fragment_list=[]# list of all non duplicate fragments to grab data for
        final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([molecule_smiles],df,duplicates,final_fragment_list,Not_found_list)
        Parent_PFAS[molecule_smiles]['Molecule_name']=name    
        for bond_idx, data in Parent_PFAS_no_duplicates[molecule_smiles].items():
            if bond_idx != 'Molecule_name':
                data['Molecule_name']=[] 
                final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([data["SMILES"][0]],df,duplicates,final_fragment_list,Not_found_list)  
                data['Molecule_name'].append(name)
                #print(name)
                final_fragment_list,name,Not_found_list=search_smiles_in_dataframe([data["SMILES"][1]],df,duplicates,final_fragment_list,Not_found_list)    
                data['Molecule_name'].append(name)
                #print(name)
    # Now collect the data
        filtered_df = df_full[df_full['Canonical_SMILES'].isin(final_fragment_list)]
        pd.set_option('display.width', None)   # auto‑detect width
        pd.set_option('display.max_columns', None)
        #print(filtered_df[filtered_df['Method']=='58'])
        # Initialize an empty dictionary to store methods and Enthalpy for each molecule

        molecule_data_hetrolytic={}

        # Iterate through the filtered DataFrame
        for _, row in filtered_df.iterrows():
            molecule = row['Canonical_SMILES']
            method = row['Method_Name']     
            method_category = row['Method_Category']
            # where is the chareg going?
            #print('This is the method:',method)
            if molecule not in PFAS_list and (any(hg in molecule for hg in list_of_head_groups) or (molecule in exact_head_groups and molecule_smiles in problem_smiles)):
                head_groups.append(molecule)
                enthalpy = row['{}'.format(head_charge)] 
                # if method =='ωB97X-V':
                #     print('THIS IS A HEAD GROUP:',molecule)
                #     print(enthalpy)
            elif molecule in PFAS_list:
                enthalpy=row['Enthalpy']
                # if method =='ωB97X-V':
                #     print(f'Whole moelcule: {molecule}')
                #     print('This is the Enthalpy of the whole molecule',enthalpy)

            else:
                tail_groups.append(molecule)
                enthalpy = row['{}'.format(tail_charge)]
                # if method =='ωB97X-V':
                #     print('THIS IS A TAIL GROUP:',molecule)
                #     print(enthalpy)
            # Only store data for molecules in final_fragment_list
            if molecule in final_fragment_list and pd.notna(enthalpy) and pd.notna(method_category):
                # If the molecule is already in the dictionary, append the method and enthalpy
                if molecule not in molecule_data_hetrolytic:
                    molecule_data_hetrolytic[molecule] = []

                # Append the method and enthalpy as a tuple
                molecule_data_hetrolytic[molecule].append((method, method_category,enthalpy))

        # Now that you have all the data do the fragments
        #print(Parent_PFAS[molecule_smiles]['Molecule_name'])
        Enthalpy_parent=enthalpy_parent(Parent_PFAS[molecule_smiles]['Molecule_name'],molecule_data_hetrolytic)

        for bond_idx, data in Parent_PFAS[molecule_smiles].items():
            if bond_idx != 'Molecule_name' and 'Molecule_name' in data:
                #print(data)
                data['BDE_hetero']={}
                Enthalpy_Data={}
                Enthalpy_Data=add_fragments(data['Molecule_name'],molecule_data_hetrolytic)

                if Enthalpy_Data and Enthalpy_parent:
                    Final_Enthalpy=calculate_enthalpy(Enthalpy_parent,Enthalpy_Data)
                    data['BDE_hetero']=Final_Enthalpy
    return Parent_PFAS,Parent_PFAS_no_duplicates
def show_image_grid(images, mols_per_row=4, sub_img_size=(2,2)): # image creation
    n = len(images)
    cols = mols_per_row
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * sub_img_size[0], rows * sub_img_size[1]))

    # Handle 1D or 2D axes
    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, img in enumerate(images):
        ax = ax_list[idx]
        ax.imshow(img)
        ax.axis("off")

    # Turn off unused axes
    for idx in range(n, rows * cols):
        ax_list[idx].axis("off")

    plt.tight_layout()
    return fig 


# In[80]:


multiple=['MRAIPLFWHJXUJN-UHFFFAOYSA-N','HCQXIHYJCFSLHP-UHFFFAOYSA-N','KWFQASBYALMRNJ-UHFFFAOYSA-N','KFUVXNKYAKCSGC-UHFFFAOYSA-M','VIGUPLPESAPZDX-UHFFFAOYSA-M']#,'SO3HCF2CF2CF2CF2CFr']#CH2rCH2CF2CF2CF2CF2CF2CF3']#'HCQXIHYJCFSLHP-UHFFFAOYSA-N']#
def load_and_clean_dataframe(pfas_type, neutral_path, anion_path, multiple, canonicalize_smiles, has_large_imag):
    Neutral_df=pd.read_csv(neutral_path)
    try:
        Anion_df = pd.read_csv(anion_path)
        combined_df = pd.concat([anion_path, neutral_path], ignore_index=True)
    except:
        combined_df=Neutral_df

    #set with alll data in it
    combined_df = combined_df.drop_duplicates(
        subset=['Molecule', 'SMILES','Method_Name'],#'SMILES', 'Method_Name'],
        keep='first',
        inplace=False  # or True if you want to modify df in place
    ).reset_index(drop=True)

    df_full = combined_df  # Update with your file name
    df_full['Canonical_SMILES'] = df_full['SMILES'].apply(canonicalize_smiles)
    # Drop duplicates based on Molecule, SMILES, Method_Name
    df=combined_df
    df = df.drop_duplicates(subset=['Molecule', 'SMILES', 'Method_Name'], keep='first').reset_index(drop=True)
    df['Canonical_SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    df=combined_df
    df['Canonical_SMILES'] = df['SMILES'].apply(canonicalize_smiles)
    df = df.loc[(df["Molecule"] != 'cis-HOCOr') & (df["Molecule"] != 'transHOCOr')] # drop these ones
    df = df[~df['Molecule'].isin(multiple)].copy() 
    df_full = df_full[~df_full['Molecule'].isin(multiple)].copy() 
    # find all the parent PFAS available
    parent_set=df[["Molecule","SMILES","Radical","Canonical_SMILES"]].drop_duplicates()
    parent_set = parent_set.loc[parent_set["Radical"] == False, ["Molecule", "Canonical_SMILES", "Radical"]].drop_duplicates()

    # duplicates #
    df=df[["Molecule","SMILES","Radical"]].drop_duplicates()
    df['Canonical_SMILES'] = df['SMILES'].apply(canonicalize_smiles)
    dupes = df[df.duplicated(subset=["Molecule", "SMILES", "Radical"], keep=False)]
    duplicates = df[df.duplicated('Canonical_SMILES', keep=False)]

    Data_Folder_Dict = {}

    for smi, group in duplicates.groupby('Canonical_SMILES'):
        molecules = group['Molecule'].tolist()
        if len(molecules) == 2:
            Data_Folder_Dict[molecules[0]] = molecules[1]
        
    #print('done')
    #print(Data_Folder_Dict)

    df_full = df_full.loc[(df_full["Molecule"] != 'cis-HOCOr') & (df_full["Molecule"] != 'transHOCOr') & (df_full["Molecule"]!='HCOOHr')& (df_full["SMILES"]!='[O-]')]
    #print(df_full[(df_full["Method"]==35) & (df_full["Molecule"]=='SO3HCF2CF2CF2CF2r')])
    mask = (
    df_full['Imaginary'].str.contains(r'\[', regex=True, na=False) &
    df_full['Imaginary'].apply(has_large_imag) &
    (df_full['Spin Contamination'] != False)
    )
    df_full = df_full[~mask].reset_index(drop=True)
    return df_full,parent_set, Data_Folder_Dict,df,duplicates


# In[81]:


# PFAS Selection. Should be updated as database increases
PFAS_OPTIONS={
"protonated": ['[H]C([H])([H])[H]','[H]C([H])([H])F','[H]C([H])(F)F','[H]C(F)(F)F','FC(F)(F)F','[H]OC(=O)C([H])([H])[H]','[H]OC(=O)C([H])([H])F','[H]OC(=O)C([H])(F)F','[H]OC(=O)C(F)(F)F',
                   '[H]OS(=O)(=O)C([H])([H])[H]','[H]OS(=O)(=O)C([H])([H])F','[H]OS(=O)(=O)C([H])(F)F','[H]OS(=O)(=O)C(F)(F)F',
                '[H]OC(=O)C(F)(F)C(F)(F)F','[H]OS(=O)(=O)C(F)(F)C(F)(F)F', # 1 CF2
                   '[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)F','[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)F', #PFBA and S equivelent #2
                   '[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',#, #PFBS and C equivelent #3
                    '[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 4 CF2
                     '[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 5 CF2
                    '[H]OC(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OS(=O)(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #4-2-FTCA and S equivelent
                   '[H]OC(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[H]OS(=O)(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #6-2-FTCA and S equivelent
                   '[H]OS(=O)(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OC(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #4-2-FTS and C equivelent
                   '[H]OS(=O)(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', '[H]OC(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',#6-2-FTS and C equivelent
                   '[H]OC(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F','[H]OS(=O)(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F', #HFPO-DA and S equivelent '[H]OC(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F',
                   '[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #PFOA and S equivelent
                   '[H]OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[H]OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',#] #PFOS and C equivelent
                    'C(Cl)(Cl)(Cl)F','C(Cl)(Cl)(F)F','C(Cl)(F)F','C(F)(F)(C(F)(F)F)','C(F)(C(F)(F)F)','C(F)(F)(F)C(=C(F))','C(F)(F)(F)(C(F)(=C))'
             ,'C(F)(F)(F)(C(=C(C(F)(F)F)))','S(F)(F)(F)(F)(F)F','C(F)(F)(F)C(F)(F)F','OCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
                  'OCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F'], "deprotonated": [ '[O-]S(=O)(=O)C([H])([H])[H]','[O-]C(=O)C([H])([H])[H]', #3H
        '[O-]S(=O)(=O)C([H])([H])F','[O-]C(=O)C([H])([H])F', # 2H
        '[O-]S(=O)(=O)C([H])(F)F','[O-]C(=O)C([H])(F)F', # 1 H
        '[O-]C(=O)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)F', # 1
        '[O-]C(=O)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)C(F)(F)F', # 2
        '[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)F', # 3 (PFBA)
        '[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',##,#]#, # 4 PFBS
        '[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 5 PFBS
        '[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 6 PFBS
        '[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #7
        '[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 8
        '[O-]C(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 4_2_FTCA and S equiv.
        '[O-]S(=O)(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]C(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', # 4_2_FTS and C equiv.
        '[O-]C(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #6_2_FTCA and S equiv.
        '[O-]C(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F','[O-]S(=O)(=O)C([H])([H])C([H])([H])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', #6_2_FTS and C equiv.
        '[O-]C(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F','[O-]S(=O)(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)']}

# Define valid PFAS types and method names


VALID_BDE_TYPES = ['homolytic', 'heterolytic']

def main():
    # This is the CSV FILES have to add in water later.... 
    neutral_csv = "https://raw.githubusercontent.com/mmarciesky/PFAS_Database/refs/heads/main/Data/Neutral_Speed.csv"
    anion_csv = "https://raw.githubusercontent.com/mmarciesky/PFAS_Database/refs/heads/main/Data/Anion_Speed.csv"
    Water_Neutral_csv="https://raw.githubusercontent.com/mmarciesky/PFAS_Database/refs/heads/main/Data/Neutral_Water_Speed.csv"
    Water_Anion_csv = "https://raw.githubusercontent.com/mmarciesky/PFAS_Database/refs/heads/main/Data/Anion_Water_Speed.csv"

    parser = argparse.ArgumentParser(
    description="Run PFAS BDE calculations for a specified PFAS type, method, and bond cleavage type."
    )

    parser.add_argument(
        '-ph', '--phase', required=False, choices=['gas', 'water'],
        help="Phase of calculation: 'gas' or 'water'"
    )
    parser.add_argument(
        '-p', '--pfas_type', required=True, choices=PFAS_OPTIONS.keys(),
        help="Choose PFAS type: 'protonated' or 'deprotonated'"
    )
    parser.add_argument(
        '-m', '--method', required=True,
        help=f"Quantum method to use. Will be validated after loading data."
    )
    parser.add_argument(
        '-t', '--bde_type', required=False, choices=VALID_BDE_TYPES,
        help="Choose bond dissociation type: 'homolytic' or 'heterolytic'"
    )
    parser.add_argument(
        '-o', '--out_file', required=False,
        help="Output file to save the image grid of BDE fragments."
    )

    args = parser.parse_args()
    if args.phase == "gas" or not args.phase:
        Neutral_csv = neutral_csv 
        Anion_csv   = anion_csv
    else: # water path fix after water is added
        Neutral_csv = neutral_csv 
        Anion_csv   = anion_csv
        print('Water path has not been added yet, defaulting to gas. :/')

    
    # pull the csv file and clean dataframe
    df_full, parent_set, Data_Folder_Dict,df,duplicates = load_and_clean_dataframe(
        pfas_type=sys.argv[sys.argv.index('-p')+1].lower() if '-p' in sys.argv else '',
        neutral_path=Neutral_csv,
        anion_path=Anion_csv,
        multiple=multiple,
        canonicalize_smiles=canonicalize_smiles,
        has_large_imag=has_large_imag
    )

    VALID_METHODS = sorted(df_full["Method_Name"].dropna().unique().tolist())

    # Extract values
    pfas_type = args.pfas_type.lower()
    method = args.method
    bde_type = args.bde_type
    
    out_file = args.out_file
    if not out_file:
        out_file='Out'
    if not pfas_type:
        pfas_type='homolytic'
    open(f"log.txt", "w").close()

    # Grab the correct PFAS list
    PFAS_list_smiles = PFAS_OPTIONS[pfas_type]
    log_mess(f"Number of PFAS Bonds: {len(PFAS_list_smiles)}")

    # Validate method again just to be safe
    if method not in VALID_METHODS:
        print(f"\n Invalid method: {method}")
        print(f" Valid methods are: {', '.join(VALID_METHODS)}")
        sys.exit(1)

    run_bde_analysis(PFAS_list_smiles, pfas_type,method, bde_type, out_file,df_full,df,duplicates)


def run_bde_analysis(smiles_list, pfas_type,method, bde_type,out_file,df_full,df,duplicates):
    log_mess(f"\nRunning BDEs for {pfas_type} PFAS using {method} and {bde_type} cleavage.")
    log_mess("Will list any fragments not available in DataBase below:")
    
    Parent_PFAS,PFAS_list=step_1_parent(smiles_list)
    Parent_PFAS,Parent_PFAS_no_duplicates=step_2(Parent_PFAS,df_full,df,duplicates,PFAS_list)
    SMILES_print_order_smiles=smiles_list
    SMILES_print_order=[] 
    for PFAS in SMILES_print_order_smiles:
        SMILES_print_order.append(canonicalize_smiles(PFAS))
    image_map = {}
    if bde_type== 'homolytic':
        BDE_type='BDE'
    else:
        BDE_type='BDE_hetero'
    BDE_type='BDE'

    for molecule_smiles in Parent_PFAS_no_duplicates:
        mol = Chem.MolFromSmiles(molecule_smiles)
        mol = Chem.AddHs(mol)
        Chem.rdDepictor.Compute2DCoords(mol)

        plotted = set()
        for bond_idx, data in Parent_PFAS_no_duplicates[molecule_smiles].items():
            if bond_idx == 'Molecule_name' or not data.get('BDE'):
                continue
            smiles_tuple = tuple(data['SMILES'])
            if smiles_tuple in plotted:
                continue

            try:
                bde = data['{}'.format(BDE_type)][method]
                mol.GetBondWithIdx(bond_idx).SetProp('bondNote', f"{bde:.2f}")
                plotted.add(smiles_tuple)
            except KeyError:
                #print(data['{}'.format(BDE_type)])
                log_mess(f"no BDE for {method} on {molecule_smiles}, bond {bond_idx}")

        # draw and store
        img = Draw.MolToImage(mol, size=(500,500), atomLabels=False)
        image_map[molecule_smiles] = img


    #print(image_map)
    ordered_images = [
        image_map[smi]
        for smi in SMILES_print_order
        if smi in image_map
    ]
    log_mess(f'This is the number of bonds available: {len(ordered_images)}')
   
    fig = show_image_grid(ordered_images)
    fig.savefig(f"{out_file}.png", dpi=300)  # Save to file
    plt.close(fig)  # optional: close to free memory


if __name__ == "__main__":
    main()





