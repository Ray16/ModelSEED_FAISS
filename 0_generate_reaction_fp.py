#!/usr/bin/env python

# input: ModelSEEDDatabase
# output: rxn_data.csv (reaction info) & reaction_embeddings.faiss (reaction indexing)

import warnings
warnings.simplefilter("ignore", SyntaxWarning)
import math
import numpy as np 
import sys
import csv
from collections import OrderedDict

sys.path.append('ModelSEEDDatabase/Libs/Python/')
from BiochemPy import Reactions, Compounds

compound_helper = Compounds()
compounds_dict = compound_helper.loadCompounds()

reaction_helper = Reactions()
reactions_dict = reaction_helper.loadReactions()

reaction_ids = list(reactions_dict.keys())

rxn_data_list = list() 
rxn_smiles_list = list()

# Output all reaction info to rxn_data.csv
for r_idx, rxn_id in enumerate(reactions_dict):

    rxn_obj = reactions_dict[rxn_id]
    
    # remove all obsolete reactions
    if rxn_obj['is_obsolete'] == 0:

        rgt_smiles_list = list()
        pdt_smiles_list = list()
        all_smiles = True

        for rgt in rxn_obj['stoichiometry']:
            cpd_id = rgt['compound']
            
            if cpd_id not in compounds_dict:
                all_smiles = False
                break
                
            cpd_smiles = compounds_dict[cpd_id]['smiles']
            
            if(cpd_smiles == ''):
                all_smiles = False
                break

            coeff = rgt['coefficient']
            count = math.ceil(abs(coeff))
            
            if(coeff < 0):
                # Reactant
                rgt_smiles_list.extend([cpd_smiles] * count)
            elif(coeff > 0):
                # Product
                pdt_smiles_list.extend([cpd_smiles] * count)

        if(all_smiles == True):
            rxn_smiles_str = '>>'.join(['.'.join(rgt_smiles_list), '.'.join(pdt_smiles_list)])
            rxn_smiles_list.append(rxn_smiles_str)
            
            # --- Collect data for CSV output ---
            rxn_data = OrderedDict()
            
            # Properties queried directly from the reaction dictionary (rxn_obj)
            rxn_data['id'] = rxn_obj.get('id', '')
            rxn_data['name'] = rxn_obj.get('name', '')
            rxn_data['abbreviation'] = rxn_obj.get('abbreviation', '')
            rxn_data['ec_numbers'] = '|'.join(rxn_obj.get('ec_numbers', [])) # Join list elements
            rxn_data['reversibility'] = rxn_obj.get('reversibility', '')
            rxn_data['deltag'] = rxn_obj.get('deltag', '')
            rxn_data['deltagerr'] = rxn_obj.get('deltagerr', '')
            rxn_data['definition'] = rxn_obj.get('definition', '')
            rxn_data['is_transport'] = rxn_obj.get('is_transport', '')
            
            # The generated property
            rxn_data['rxn_smiles'] = rxn_smiles_str
            
            rxn_data_list.append(rxn_data)

print(f'\nGenerated SMILES string for {len(rxn_smiles_list)} reactions.')

csv_file = 'rxn_data.csv'

if not rxn_data_list:
    print("No data to write to CSV.")
    sys.exit(0)

fieldnames = list(rxn_data_list[0].keys())

print(f'Writing data to {csv_file}...')

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(rxn_data_list)

    print(f'Successfully created {csv_file} with {len(rxn_data_list)} reaction entries.')

except Exception as e:
    print(f"An error occurred while writing to CSV: {e}")


# Generate fingerprints for all reactions
from rxnfp.transformer_fingerprints import (
RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
rxn_fps = list()
for i in range(0, len(rxn_smiles_list), 1000):
    print("Chunk: ",i)
    chunk = rxn_smiles_list[i:i + 1000]
    tmp_fps = rxnfp_generator.convert_batch(chunk)
    rxn_fps.extend(tmp_fps)
    print(len(rxn_fps), len(rxn_fps[0]))

    rxn_fps_array = np.array(rxn_fps, dtype=np.float32)
    np.save('rxn_fingerprints.npy', rxn_fps_array)

    print('Generated the fingerprints for all reactions.')