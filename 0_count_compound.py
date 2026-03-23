#!/usr/bin/env python

# input: ModelSEEDDatabase
# output: cpd_freq.csv (compound frequency ranking)

import warnings
warnings.simplefilter("ignore", SyntaxWarning)
import sys
import pandas as pd
from collections import Counter

sys.path.append('ModelSEEDDatabase/Libs/Python/')
from BiochemPy import Reactions, Compounds

compound_helper = Compounds()
compounds_dict = compound_helper.loadCompounds()

reaction_helper = Reactions()
reactions_dict = reaction_helper.loadReactions()

cpd_counter = Counter()

for rxn_id, rxn_obj in reactions_dict.items():
    if rxn_obj['is_obsolete'] == 0:
        for rgt in rxn_obj['stoichiometry']:
            cpd_id = rgt['compound']
            cpd_counter[cpd_id] += 1

# Build dataframe sorted by decreasing frequency
df = pd.DataFrame(cpd_counter.most_common(), columns=['compound_id', 'frequency'])
df['name'] = df['compound_id'].map(lambda cid: compounds_dict[cid]['name'] if cid in compounds_dict else '')
df.to_csv('cpd_freq.csv', index=False)

print(f'Counted {len(cpd_counter)} unique compounds across non-obsolete reactions.')
print(f'Saved to cpd_freq.csv')
print(df.head(20))
