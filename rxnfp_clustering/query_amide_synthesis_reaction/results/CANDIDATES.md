# Candidates worth a bench slot

24 proteins, each supported by at least 0 independent observations that are not merely family membership. Ordered by how many lines agree, then by substrate distance.

Entries marked a NEAR MISS did not reach the convergence threshold and are included on a separate, stated criterion: two independent lines plus demonstrated room for both monomers. The threshold was not moved to accommodate them.

**None of these has been assayed on nylon-6,6.** Convergence means several methods that fail differently happen to agree, which is a reason to spend a bench slot and not evidence that the enzyme works. With three measured positives in the reference panel, no ranking here can be validated.

---

## Q81RQ8 — Citryl-spermidine/3,4-dihydroxybenzoyl-citryl-spermidine:spermidine ligase (EC 6.3.2.-) (Petrobactin biosynthesis protei

**Organism** Bacillus anthracis  
**Gene** `asbB GBAA_1982`  
**Length** 612.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.789** (own_curated_reaction)
- AlphaFold pLDDT **96.1**
- structure `3TO3` at 2.382 Å
- in-line adenylation geometry 5.2 Å

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 23 between amines, against HMDA's 6 — **room for the acceptor**
- widest acid span 2, short of adipate's 4

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=N(8)-citryl-spermidine + spermidine + ATP = N(8),N'(8)-citryl-bis(spermidine) + AMP + diphosphate + H(+); Xref=Rhea:RHEA:63812, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57834, ChEBI:CHEBI:149586, ChEBI:CHEBI:149592, ChEBI:CHEBI:456215; Evidence=; CATALYTIC ACTIVITY: Reaction=N(1)-(3,4-dihydroxybenzoyl)-N(8)-citryl-spermidine + spermidine + A

**Function** FUNCTION: Involved in the biosynthesis of petrobactin, a catecholate siderophore that functions in both iron acquisition and virulence (PubMed:16875672, PubMed:17189355, PubMed:22408253). Catalyzes the ATP-dependent condensation of spermidine with N(8)-citryl-spermidine or N(1)-(3,4-dihydroxbenzoyl)-N(8)-citryl-spermidine, two intermediates in petrobactin biosynthesis pathway (PubMed:18758617, Pub

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**Mutagenesis** (residue-level functional data)

- 313: K->A: Preference for spermidine is not completely abrogated, but activity is increased with unnatural nucleophiles such as spermine, l,8-diaminooctane and 
- 313: K->M: Increases acceptance of polyamines lacking a secondary amine.
- 461: E->A: Increases relative acceptance of spermine, norspermidine and 1-aminoethyl-1,3-propanediamine.
- 461: E->D: Displays selectivity close to that of wild type.

**Literature** [18952800](https://pubmed.ncbi.nlm.nih.gov/18952800/), [14756782](https://pubmed.ncbi.nlm.nih.gov/14756782/), [16875672](https://pubmed.ncbi.nlm.nih.gov/16875672/), [17189355](https://pubmed.ncbi.nlm.nih.gov/17189355/), [18758617](https://pubmed.ncbi.nlm.nih.gov/18758617/), [27425635](https://pubmed.ncbi.nlm.nih.gov/27425635/), [22408253](https://pubmed.ncbi.nlm.nih.gov/22408253/)

## P9WJL3 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)  
**Gene** `murE Rv2158c MTCY270.10`  
**Length** 535.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **91.5**
- structure `2XJA` at 3.0 Å
- in-line adenylation geometry 6.2 Å
- crystallographic ligand `UAG`
- curated ligand `CHEBI:57791`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 5, short of HMDA's 6
- spans 10 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

**Note** MISCELLANEOUS: Was identified as a high-confidence drug target.

**BRENDA** `6.3.2.13;` — an entry exists; the API needs credentials this project does not have

**Literature** [9634230](https://pubmed.ncbi.nlm.nih.gov/9634230/), [19099550](https://pubmed.ncbi.nlm.nih.gov/19099550/), [21969609](https://pubmed.ncbi.nlm.nih.gov/21969609/)

## Q46E79 — 3-methyl-D-ornithine--L-lysine ligase (EC 6.3.2.59) ((3R)-3-methyl-D-ornithine:L-lysine ligase) (Pyrrolysine biosynthesi

**Organism** Methanosarcina barkeri (strain Fusaro / DSM 804)  
**Gene** `pylC Mbar_A0837`  
**Length** 363.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.667** (own_modelseed_reaction)
- its EC class reaches 0.667
- AlphaFold pLDDT **94.8**
- structure `4FFL` at 1.5 Å
- in-line adenylation geometry 6.61 Å
- crystallographic ligand `0TK`
- curated ligand `CHEBI:32551;CHEBI:57668`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 11 between amines, against HMDA's 6 — **room for the acceptor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(3R)-3-methyl-D-ornithine + L-lysine + ATP = (3R)-3-methyl-D-ornithyl-N(6)-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:32763, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:64642, ChEBI:CHEBI:64643, ChEBI:CHEBI:456216; EC=6.3.2.59; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:32764; Evidence=

**Function** FUNCTION: Is required for the biosynthesis of pyrrolysine (By similarity). Catalyzes the ATP-dependent ligation between (3R)-3-methyl-D-ornithine and L-lysine, leading to (3R)-3-methyl-D-ornithyl-N6-L-lysine (Probable). .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 magnesium ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the PylC family. .

**BRENDA** `6.3.2.59;` — an entry exists; the API needs credentials this project does not have

**Literature** [16980466](https://pubmed.ncbi.nlm.nih.gov/16980466/), [22985965](https://pubmed.ncbi.nlm.nih.gov/22985965/)

## Q9RDS6 — Gamma-glutamylpolyamine synthetase GlnA2 (EC 6.3.1.-)

**Organism** Streptomyces coelicolor (strain ATCC BAA-471 / A3(2) / M145)  
**Gene** `glnA2 SCO2241`  
**Length** 453.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.873** (own_curated_reaction)
- AlphaFold pLDDT **92.7**
- measured K_M **0.13 mM**
- curated ligand `CHEBI:29985`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 17 between amines, against HMDA's 6 — **room for the acceptor**
- widest acid span 3, short of adipate's 4

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `cadaverine` — C5 diamine - ONE methylene short of HMDA
- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=; CATALYTIC ACTIVITY: Reaction=spermine + L-glut

**Function** FUNCTION: Involved in the catabolism of polyamines. Catalyzes the ATP-dependent gamma-glutamylation of polyamines. Substrates include putrescine, cadaverine, spermidine and spermine, with a preference for short-chain polyamine putrescine (PubMed:35409114). No complementation of the L-glutamine auxotrophy of an E.coli glnA mutant (PubMed:16932908). Together with GlnA3, enables survival of S.coelico

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

**Biotechnology** BIOTECHNOLOGY: This protein has some activity with cysteine and hence may be useful in biotechnical applications as gamma-glutamylation of cysteine strongly increases its solubility in water, which is critical for its usage in medications. .

**Literature** [12000953](https://pubmed.ncbi.nlm.nih.gov/12000953/), [16932908](https://pubmed.ncbi.nlm.nih.gov/16932908/), [28487688](https://pubmed.ncbi.nlm.nih.gov/28487688/), [35409114](https://pubmed.ncbi.nlm.nih.gov/35409114/)

## Q2FWC5 — L-aspartate--L-methionine ligase (LdmS) (EC 6.3.2.-) (ATP-dependent dipeptide ligase) (ATP-grasp enzyme) (L-Asp-L-Met li

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `SAOUHSC_02373`  
**Length** 397.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.869** (own_modelseed_reaction)
- AlphaFold pLDDT **94.2**
- structure `7R8P` at 1.37 Å
- in-line adenylation geometry 4.0 Å
- measured K_M **0.31 mM**
- crystallographic ligand `CIT`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 5 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=L-aspartate + L-methionine + ATP = L-aspartyl-L-methionine + ADP + phosphate + H(+); Xref=Rhea:RHEA:76447, ChEBI:CHEBI:15378, ChEBI:CHEBI:29991, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57844, ChEBI:CHEBI:195235, ChEBI:CHEBI:456216; Evidence=

**Function** FUNCTION: L-amino acid ligase, which preferentially catalyzes the formation of L-aspartyl-L-methionine dipeptide from L-aspartate and L-methionine in the presence of ATP. Less active with L-asparagine and L-methionine as substrates. Less active with L-aspartate and either L-phenylalanine, L-valine, L-leucine or L-isoleucine as substrates. Decreased activity when L-methionine is substituted with se

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Mutagenesis** (residue-level functional data)

- 44: L->A: 19% relative activity.
- 44: L->E,D: Loss of activity.
- 44: L->F: 14% relative activity.
- 44: L->H: 13% relative activity.
- 44: L->K: 7% relative activity.
- 44: L->Y: 8% relative activity.
- 47: R->A,H: Loss of activity.
- 47: R->K: 9% relative activity.

**Literature** [25888466](https://pubmed.ncbi.nlm.nih.gov/25888466/), [35988643](https://pubmed.ncbi.nlm.nih.gov/35988643/)

## P78061 — Gamma-glutamylputrescine synthetase PuuA (Gamma-Glu-Put synthetase) (EC 6.3.1.11) (Glutamate--putrescine ligase)

**Organism** Escherichia coli (strain K12)  
**Gene** `puuA ycjK b1297 JW5201`  
**Length** 472.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.857** (own_modelseed_reaction)
- its EC class reaches 0.857
- AlphaFold pLDDT **92.7**
- measured K_M **2.07 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 9 between amines, against HMDA's 6 — **room for the acceptor**
- widest acid span 3, short of adipate's 4

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; EC=6.3.1.11; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=

**Function** FUNCTION: Involved in the breakdown of putrescine. Catalyzes the ATP-dependent gamma-glutamylation of putrescine, producing gamma-L-glutamylputrescine. Absolutely essential to utilize putrescine as both nitrogen and carbon sources and to decrease the toxicity of putrescine, which can lead to inhibition of cell growth and protein synthesis. In vitro is also able to use several diamines, and spermid

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Name=Mn(2+); Xref=ChEBI:CHEBI:29035; Evidence=; Note=Requires Mg(2+), that can be substituted by Mn(2+).

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

**Mutagenesis** (residue-level functional data)

- 282: H->N: Activity is impaired to 9% of wild-type.
- 357: R->Q: Activity is impaired to 3% of wild-type.

**Literature** [9097039](https://pubmed.ncbi.nlm.nih.gov/9097039/), [9278503](https://pubmed.ncbi.nlm.nih.gov/9278503/), [16738553](https://pubmed.ncbi.nlm.nih.gov/16738553/), [14527658](https://pubmed.ncbi.nlm.nih.gov/14527658/), [15590624](https://pubmed.ncbi.nlm.nih.gov/15590624/), [18495664](https://pubmed.ncbi.nlm.nih.gov/18495664/)

## E2JA32 — Fumarate--(S)-2,3-diaminopropanoate ligase (EC 6.3.2.46) (ATP-dependent fumaroyl-DAP ligase)

**Organism** Enterobacter agglomerans (Erwinia herbicola) (Pantoea agglomerans)  
**Gene** `ddaG`  
**Length** 396.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.848** (own_modelseed_reaction)
- its EC class reaches 0.848
- AlphaFold pLDDT **88.3**
- measured K_M **0.55 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 2, short of HMDA's 6
- spans 6 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(S)-2,3-diaminopropanoate + fumarate + ATP = N(3)-fumaroyl-(S)-2,3-diaminopropanoate + AMP + diphosphate; Xref=Rhea:RHEA:44340, ChEBI:CHEBI:29806, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:84330, ChEBI:CHEBI:456215; EC=6.3.2.46; Evidence=

**Function** FUNCTION: Involved in dapdiamide antibiotics biosynthesis. Ligates fumarate and 2,3-diaminopropionate (DAP) to form N-beta-fumaroyl-DAP. Can also form N-succinoyl-DAP from succinate and DAP, with lower efficiency. .

**BRENDA** `6.3.2.46;` — an entry exists; the API needs credentials this project does not have

**Literature** [20945916](https://pubmed.ncbi.nlm.nih.gov/20945916/), [19807062](https://pubmed.ncbi.nlm.nih.gov/19807062/)

## Q2FW70 — D-ornithine--citrate ligase (EC 6.3.2.58)

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sfnaD SAOUHSC_02436`  
**Length** 658.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.809** (own_modelseed_reaction)
- its EC class reaches 0.809
- AlphaFold pLDDT **94.4**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 4, short of HMDA's 6
- spans 9 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=D-ornithine + citrate + ATP = N(5)-[(S)-citryl]-D-ornithine + AMP + diphosphate + H(+); Xref=Rhea:RHEA:59136, ChEBI:CHEBI:15378, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57668, ChEBI:CHEBI:142972, ChEBI:CHEBI:456215; EC=6.3.2.58; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:59137; Evidence=

**Function** FUNCTION: Involved in the biosynthesis of the siderophore staphyloferrin A. Catalyzes the ATP-dependent condensation of D-ornithine and citrate to form a citryl-D-ornithine intermediate. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**Literature** [19138128](https://pubmed.ncbi.nlm.nih.gov/19138128/)

## Q2G1M8 — 2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate synthase (EC 6.3.2.55) (Staphyloferrin 

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sbnF SAOUHSC_00080`  
**Length** 616.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.807** (own_curated_reaction)
- its EC class reaches 0.761
- AlphaFold pLDDT **91.1**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 11 between amines, against HMDA's 6 — **room for the acceptor**
- spans 6 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=2-[(2-aminoethylcarbamoyl)methyl]-2-hydroxybutanedioate + (S)-2,3-diaminopropanoate + ATP = 2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate + AMP + diphosphate; Xref=Rhea:RHEA:59128, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:142970, ChEBI:CHEBI:142971, ChEBI:CHEBI:456215; EC=6.3.2.55; Evidence=; PhysiologicalD

**Function** FUNCTION: Catalyzes the condensation of L-2,3-diaminopropionic acid (L-Dap) and citryl-diaminoethane to form L-2,3-diaminopropionyl-citryl-diaminoethane, the third step in staphyloferrin B biosynthesis. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**BRENDA** `6.3.2.55;` — an entry exists; the API needs credentials this project does not have

**Literature** [14688077](https://pubmed.ncbi.nlm.nih.gov/14688077/), [19775248](https://pubmed.ncbi.nlm.nih.gov/19775248/)

## Q2G1M9 — L-2,3-diaminopropanoate--citrate ligase (EC 6.3.2.54) (Staphyloferrin B biosynthesis protein SbnE)

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sbnE SAOUHSC_00079`  
**Length** 578.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.807** (own_modelseed_reaction)
- its EC class reaches 0.807
- AlphaFold pLDDT **92.6**
- measured K_M **0.99 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 2, short of HMDA's 6
- spans 7 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(S)-2,3-diaminopropanoate + citrate + ATP = 2-[(L-alanin-3-ylcarbamoyl)methyl]-2-hydroxybutanedioate + AMP + diphosphate; Xref=Rhea:RHEA:59124, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:142969, ChEBI:CHEBI:456215; EC=6.3.2.54; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:59125; Evidence=

**Function** FUNCTION: Catalyzes the synthesis of citryl-L-2,3-diaminopropionic acid from L-2,3-diaminopropionic acid (L-Dap) and citrate, the first step in staphyloferrin B biosynthesis. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**BRENDA** `6.3.2.54;` — an entry exists; the API needs credentials this project does not have

**Literature** [14688077](https://pubmed.ncbi.nlm.nih.gov/14688077/), [19775248](https://pubmed.ncbi.nlm.nih.gov/19775248/)

## A0A0F7RJ52 — Spermidine-citrate ligase (EC 6.3.2.-) (Petrobactin biosynthesis protein AsbA)

**Organism** Bacillus anthracis  
**Gene** `asbA GBAA_1981`  
**Length** 602.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.789** (own_modelseed_reaction)
- AlphaFold pLDDT **93.4**
- measured K_M **5.65 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 8 between amines, against HMDA's 6 — **room for the acceptor**
- widest acid span 3, short of adipate's 4

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + citrate + ATP = N(8)-citryl-spermidine + AMP + diphosphate + H(+); Xref=Rhea:RHEA:63808, ChEBI:CHEBI:15378, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57834, ChEBI:CHEBI:149586, ChEBI:CHEBI:456215; Evidence=

**Function** FUNCTION: Involved in the biosynthesis of petrobactin, a catecholate siderophore that functions in both iron acquisition and virulence (PubMed:16875672, PubMed:17189355, PubMed:22408253). Catalyzes the ATP-dependent condensation of citric acid and spermidine to form N(8)-citryl-spermidine (PubMed:17579415, PubMed:19259597, PubMed:22408253). It can also catalyze the condensation of several di- and 

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**Biotechnology** BIOTECHNOLOGY: The ability of the enzyme to catalyze condensation of citric acid with several di- and triamine analogs of spermidine suggests that it may be a versatile biocatalyst for the preparation of highly enantiomerically-enriched citrate derivatives via a novel type of desymmetrization process. .

**Literature** [18952800](https://pubmed.ncbi.nlm.nih.gov/18952800/), [14756782](https://pubmed.ncbi.nlm.nih.gov/14756782/), [16875672](https://pubmed.ncbi.nlm.nih.gov/16875672/), [17189355](https://pubmed.ncbi.nlm.nih.gov/17189355/), [17579415](https://pubmed.ncbi.nlm.nih.gov/17579415/), [19259597](https://pubmed.ncbi.nlm.nih.gov/19259597/), [22408253](https://pubmed.ncbi.nlm.nih.gov/22408253/), [27425635](https://pubmed.ncbi.nlm.nih.gov/27425635/)

## Q2FZP6 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--L-lysine ligase (EC 6.3.2.7) (L-lysine-adding enzyme) (UDP-MurNAc-L-Ala-D-Glu

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `murE SAOUHSC_00954`  
**Length** 493.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure

**Resolved**

- substrate distance **0.784** (own_modelseed_reaction)
- its EC class reaches 0.784
- AlphaFold pLDDT **96.6**
- structure `4C12` at 1.8 Å
- in-line adenylation geometry 6.51 Å
- measured K_M **0.087 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 5, short of HMDA's 6
- spans 6 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + L-lysine + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:17969, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:83900, ChEBI:CHEBI:83903, ChEBI:CHEBI:456216; EC=6.3.2.7; Evidence=

**Function** FUNCTION: Catalyzes the addition of L-lysine to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Cannot use diaminopimelate as substrate (PubMed:10498701, PubMed:14114846, PubMed:20659527). Can accept L-ornithine as substrate, but the efficiency is 400-fold lower than that with L-lysine (PubMed:20659527). Seems to h

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

**Note** MISCELLANEOUS: Expression of this protein in E.coli results in 50% incorporation of L-Lys into the peptidoglycan (which normally contains meso-diaminopimelate instead), leading to abnormal morphological changes and subsequent cell lysis. .

**BRENDA** `6.3.2.7;` — an entry exists; the API needs credentials this project does not have

**Literature** [14114846](https://pubmed.ncbi.nlm.nih.gov/14114846/), [10498701](https://pubmed.ncbi.nlm.nih.gov/10498701/), [14996801](https://pubmed.ncbi.nlm.nih.gov/14996801/), [20659527](https://pubmed.ncbi.nlm.nih.gov/20659527/)

## Q9WY79 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--LD-lysine ligase (EC 6.3.2.37) (EC 6.3.2.7) (D-lysine-adding enzyme) (L-lysin

**Organism** Thermotoga maritima (strain ATCC 43589 / DSM 3109 / JCM 10099 / NBRC 100826 / MSB8)  
**Gene** `murE TM_0237`  
**Length** 490.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure

**Resolved**

- substrate distance **0.784** (own_modelseed_reaction)
- its EC class reaches 0.784
- AlphaFold pLDDT **94.5**
- structure `4BUB` at 2.9 Å
- in-line adenylation geometry 6.37 Å
- measured K_M **0.45 mM**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 5, short of HMDA's 6
- spans 10 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + L-lysine + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:17969, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:83900, ChEBI:CHEBI:83903, ChEBI:CHEBI:456216; EC=6.3.2.7; Evidence=; CATALYTIC ACTIVITY: Reaction=UD

**Function** FUNCTION: Catalyzes the addition of both L- and D-lysine to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Is also able to use meso-diaminopimelate as the amino acid substrate in vitro, although much less efficiently. .

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

**Note** MISCELLANEOUS: The peptidoglycan of T.maritima was shown to contain approximate amounts of both enantiomers of lysine and no diaminopimelate.

**Literature** [10360571](https://pubmed.ncbi.nlm.nih.gov/10360571/), [16595662](https://pubmed.ncbi.nlm.nih.gov/16595662/)

## P22188 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Escherichia coli (strain K12)  
**Gene** `murE b0085 JW0083`  
**Length** 495.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **96.2**
- structure `7B61` at 1.65 Å
- measured K_M **0.036 mM**
- crystallographic ligand `API;UAG`
- curated ligand `CHEBI:57791`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- widest amine span 5, short of HMDA's 6
- spans 10 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Is also able to use many meso-diaminopimelate analogs as substrates, although much less efficiently, but not L-lysine. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

**BRENDA** `6.3.2.13;` — an entry exists; the API needs credentials this project does not have

**Literature** [2692800](https://pubmed.ncbi.nlm.nih.gov/2692800/), [1630901](https://pubmed.ncbi.nlm.nih.gov/1630901/), [9278503](https://pubmed.ncbi.nlm.nih.gov/9278503/), [16738553](https://pubmed.ncbi.nlm.nih.gov/16738553/), [9166795](https://pubmed.ncbi.nlm.nih.gov/9166795/), [2198024](https://pubmed.ncbi.nlm.nih.gov/2198024/), [2269304](https://pubmed.ncbi.nlm.nih.gov/2269304/), [3905407](https://pubmed.ncbi.nlm.nih.gov/3905407/)

## Q2G1N1 — Staphyloferrin B synthase (EC 6.3.2.56) (Staphyloferrin B biosynthesis protein SbnC)

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sbnC SAOUHSC_00077`  
**Length** 584.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.761** (own_curated_reaction)
- its EC class reaches 0.758
- AlphaFold pLDDT **96.4**
- structure `7CBB` at 2.6 Å

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 11 between amines, against HMDA's 6 — **room for the acceptor**
- spans 16 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate + 2-oxoglutarate + ATP = staphyloferrin B + AMP + diphosphate + H(+); Xref=Rhea:RHEA:59132, ChEBI:CHEBI:15378, ChEBI:CHEBI:16810, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:136993, ChEBI:CHEBI:142971, ChEBI:CHEBI:456215; EC=6.3.2.56; Evidence=; PhysiologicalDirection=left-to-right; 

**Function** FUNCTION: Catalyzes the condensation of L-2,3-diaminopropionyl-citryl-diaminoethane and 2-oxoglutarate to form staphyloferrin B, the fourth and last step in staphyloferrin B biosynthesis. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**BRENDA** `6.3.2.56;` — an entry exists; the API needs credentials this project does not have

**Literature** [14688077](https://pubmed.ncbi.nlm.nih.gov/14688077/), [19775248](https://pubmed.ncbi.nlm.nih.gov/19775248/)

## P0AES0 — Bifunctional glutathionylspermidine synthetase/amidase (GspSA) [Includes: Glutathionylspermidine amidase (Gsp amidase) (

**Organism** Escherichia coli (strain K12)  
**Gene** `gss gsp b2988 JW2956`  
**Length** 619.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.738** (own_modelseed_reaction)
- its EC class reaches 0.738
- AlphaFold pLDDT **95.4**
- structure `2IO8` at 2.1 Å
- in-line adenylation geometry 3.54 Å
- measured K_M **0.02 mM**
- crystallographic ligand `GGA;GSH`
- curated ligand `CHEBI:57834;CHEBI:57835;CHEBI:57925`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 19 between amines, against HMDA's 6 — **room for the acceptor**
- spans 9 between acids, against adipate's 4 — **room for the donor**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+); Xref=Rhea:RHEA:21272, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57834, ChEBI:CHEBI:57835, ChEBI:CHEBI:57925, ChEBI:CHEBI:456216; EC=6.3.1.8; Evidence=; CATALYTIC ACTIVITY: Reaction=glutathionylspermidine + H2O = spermidine + glutathione; Xref=Rhea:RHEA:17173, Ch

**Function** FUNCTION: Catalyzes the formation of an amide bond between glutathione (GSH) and spermidine coupled with hydrolysis of ATP; also catalyzes the opposing reaction, i.e. the hydrolysis of glutathionylspermidine (Gsp) back to glutathione and spermidine. The amidase active site can also hydrolyze Gsp-disulfide (Gsp-S-S-Gsp) to Gsp-SG and Gsp S-thiolated proteins (GspSSPs) to GSH S-thiolated protein (GS

**Family (UniProt)** SIMILARITY: In the C-terminal section; belongs to the glutathionylspermidine synthase preATP-grasp family. .

**Note** MISCELLANEOUS: Gsp forms mixed disulfides with the thiols of a variety of E.coli proteins. These mixed disulfides represent a previously uncharacterized type of post-translational modification. The level of these proteins is increased by oxidative stress, which implies that Gsp might protect protein thiols against irreversible oxidation (PubMed:20530482). .; MISCELLANEOUS: No metal ion is required

**Mutagenesis** (residue-level functional data)

- 59: C->A: Loss of amidase activity.
- 173: C->A: No effect on amidase activity.
- 316: R->E: Loss of synthetase activity.
- 335: S->A: 3.6-fold decrease in GSH affinity, 1.6-fold decrease in spermidine activity, and 1.3-fold decrease in synthetase activity.
- 337: S->A: No effect on GSH and spermidine affinity, but 2-fold decrease in synthetase activity.
- 338: C->A: 10-fold decrease in GSH affinity, 5-fold decrease in spermidine activity, but no effect on synthetase activity.
- 391: E->A: 2-fold decrease in GSH affinity, 60-fold decrease in spermidine activity, and 10-fold decrease in synthetase activity.
- 392: E->A: 33-fold decrease in GSH affinity, 13-fold decrease in spermidine activity, and 6-fold decrease in synthetase activity.

**BRENDA** `3.5.1.78;6.3.1.8;` — an entry exists; the API needs credentials this project does not have

**Literature** [7775463](https://pubmed.ncbi.nlm.nih.gov/7775463/), [9278503](https://pubmed.ncbi.nlm.nih.gov/9278503/), [16738553](https://pubmed.ncbi.nlm.nih.gov/16738553/), [9398217](https://pubmed.ncbi.nlm.nih.gov/9398217/), [8999955](https://pubmed.ncbi.nlm.nih.gov/8999955/), [23097746](https://pubmed.ncbi.nlm.nih.gov/23097746/), [17124497](https://pubmed.ncbi.nlm.nih.gov/17124497/), [20530482](https://pubmed.ncbi.nlm.nih.gov/20530482/)

## P90518 — Glutathionylspermidine synthase (EC 6.3.1.8)

**Organism** Crithidia fasciculata  
**Gene** `GSP`  
**Length** 719.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, lead_family_verdict, bifunctional_ligand_in_the_pocket, independent_geometric_score

**Resolved**

- substrate distance **0.738** (own_modelseed_reaction)
- its EC class reaches 0.738
- AlphaFold pLDDT **88.2**
- curated ligand `CHEBI:57834;CHEBI:57925`

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 19 between amines, against HMDA's 6 — **room for the acceptor**
- spans 9 between acids, against adipate's 4 — **room for the donor**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+); Xref=Rhea:RHEA:21272, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57834, ChEBI:CHEBI:57835, ChEBI:CHEBI:57925, ChEBI:CHEBI:456216; EC=6.3.1.8

**Function** FUNCTION: Conjugates glutathione (gamma-Glu-Cys-Gly) and spermidine to form glutathionylspermidine in the biosynthesis trypanothione (N(1),N(8)-bis(glutathionyl)spermidine), which is involved in maintaining intracellular thiol redox and in defense against oxidants.

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420

**Family (UniProt)** SIMILARITY: In the C-terminal section; belongs to the glutathionylspermidine synthase preATP-grasp family. .

**BRENDA** `6.3.1.8;` — an entry exists; the API needs credentials this project does not have

**Literature** [9677355](https://pubmed.ncbi.nlm.nih.gov/9677355/), [1304372](https://pubmed.ncbi.nlm.nih.gov/1304372/)

## O60993 — Trypanothione synthetase (EC 6.3.1.9) (Cf-TS)

**Organism** Crithidia fasciculata  
**Gene** `TRS`  
**Length** 652.0 aa  

**Why it is here** — a NEAR MISS: it does not reach the convergence threshold, but its pocket has demonstrably held both monomers' worth of span. Admitted on that stated criterion, not by lowering the floor.

**Resolved**

- its EC class reaches 0.738
- AlphaFold pLDDT **93.2**

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 30 between amines, against HMDA's 6 — **room for the acceptor**
- spans 30 between acids, against adipate's 4 — **room for the donor**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+); Xref=Rhea:RHEA:21272, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57834, ChEBI:CHEBI:57835, ChEBI:CHEBI:57925, ChEBI:CHEBI:456216; EC=6.3.1.9; CATALYTIC ACTIVITY: Reaction=glutathionylspermidine + glutathione + ATP = trypanothione + ADP + phosphate + H(+); Xref=Rh

**Function** FUNCTION: Conjugates glutathione (gamma-Glu-Cys-Gly) and glutathionylspermidine to form trypanothione (N(1),N(8)-bis(glutathionyl)spermidine), which is involved in maintaining intracellular thiol redox and in defense against oxidants.

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420

**Family (UniProt)** SIMILARITY: In the C-terminal section; belongs to the glutathionylspermidine synthase preATP-grasp family. .

**BRENDA** `6.3.1.9;` — an entry exists; the API needs credentials this project does not have

**Literature** [9677355](https://pubmed.ncbi.nlm.nih.gov/9677355/), [1304372](https://pubmed.ncbi.nlm.nih.gov/1304372/), [9115252](https://pubmed.ncbi.nlm.nih.gov/9115252/)

## Q113V7 — Cyanophycin synthetase (EC 6.3.2.29) (EC 6.3.2.30) (Cyanophycin synthase)

**Organism** Trichodesmium erythraeum (strain IMS101)  
**Gene** `Tery_1965`  
**Length** 902.0 aa  

**Why it is here** — a NEAR MISS: it does not reach the convergence threshold, but its pocket has demonstrably held both monomers' worth of span. Admitted on that stated criterion, not by lowering the floor.

**Resolved**

- AlphaFold pLDDT **90.6**
- structure `7WAF` at 2.52 Å

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 21 between amines, against HMDA's 6 — **room for the acceptor**
- spans 15 between acids, against adipate's 4 — **room for the donor**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=[L-4-(L-arginin-2-N-yl)aspartate](n) + L-aspartate + ATP = [L-4-(L-arginin-2-N-yl)aspartate](n)-L-aspartate + ADP + phosphate + H(+); Xref=Rhea:RHEA:13277, Rhea:RHEA-COMP:13728, Rhea:RHEA-COMP:13733, ChEBI:CHEBI:15378, ChEBI:CHEBI:29991, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:137986, ChEBI:CHEBI:137990, ChEBI:CHEBI:456216; EC=6.3.2.29; Evidence=; CATALYTIC A

**Function** FUNCTION: Catalyzes the ATP-dependent polymerization of arginine and aspartate to multi-L-arginyl-poly-L-aspartic acid (cyanophycin; a water-insoluble reserve polymer). .

**Family (UniProt)** SIMILARITY: In the C-terminal section; belongs to the MurCDEF family. .

**BRENDA** `6.3.2.29;` — an entry exists; the API needs credentials this project does not have

**Literature** [36042318](https://pubmed.ncbi.nlm.nih.gov/36042318/)

## P0AES1 — Bifunctional glutathionylspermidine synthetase/amidase (GspSA) [Includes: Glutathionylspermidine amidase (Gsp amidase) (

**Organism** Shigella flexneri  
**Gene** `gss gsp SF3035 S3236`  
**Length** 619.0 aa  

**Why it is here** — a NEAR MISS: it does not reach the convergence threshold, but its pocket has demonstrably held both monomers' worth of span. Admitted on that stated criterion, not by lowering the floor.

**Resolved**

- its EC class reaches 0.738

**Pocket capacity** (from ligands it is curated to handle; room, not competence)

- spans 19 between amines, against HMDA's 6 — **room for the acceptor**
- spans 9 between acids, against adipate's 4 — **room for the donor**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+); Xref=Rhea:RHEA:21272, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57834, ChEBI:CHEBI:57835, ChEBI:CHEBI:57925, ChEBI:CHEBI:456216; EC=6.3.1.8; Evidence=; CATALYTIC ACTIVITY: Reaction=glutathionylspermidine + H2O = spermidine + glutathione; Xref=Rhea:RHEA:17173, Ch

**Function** FUNCTION: Catalyzes the formation of an amide bond between glutathione (GSH) and spermidine coupled with hydrolysis of ATP; also catalyzes the opposing reaction, i.e. the hydrolysis of glutathionylspermidine (Gsp) back to glutathione and spermidine. May act synergistically with glutaredoxin to regulate the redox environment and defend against oxidative damage. .

**Family (UniProt)** SIMILARITY: In the C-terminal section; belongs to the glutathionylspermidine synthase preATP-grasp family. .

**Literature** [12384590](https://pubmed.ncbi.nlm.nih.gov/12384590/), [12704152](https://pubmed.ncbi.nlm.nih.gov/12704152/)

## D2TMN3 — Enterobactin synthetase component E (EC 2.3.1.-, EC 2.7.7.58)

**Organism** Citrobacter rodentium (strain ICC168) (Citrobacter freundii biotype 4280)  
**Gene** `entE ROD_06031`  
**Length** 535.0 aa  

**Why it is here** — the OTHER hypothesis. It reaches no independent evidence line in this pipeline; it is here because it is among the ANL candidates most similar to a protein with MEASURED activity on the nylon-6,6 diad: 30.2% identity to AS9, a measured producer.

**Literature** [19897651](https://pubmed.ncbi.nlm.nih.gov/19897651/)

## A0A075UYU2 — Amide synthetase

**Organism** Amycolatopsis japonica  
**Gene** `AJAP_12390`  
**Length** 521.0 aa  

**Why it is here** — the OTHER hypothesis. It reaches no independent evidence line in this pipeline; it is here because it is among the ANL candidates most similar to a protein with MEASURED activity on the nylon-6,6 diad: 30.6% identity to AS9, a measured producer.

**Family (UniProt)** SIMILARITY: Belongs to the ATP-dependent AMP-binding enzyme family. .

**Literature** [25193710](https://pubmed.ncbi.nlm.nih.gov/25193710/)

## A0ABC8C1M7 — Amide synthetase

**Organism** Kitasatospora albolonga  
**Gene** `B7C62_33055`  
**Length** 514.0 aa  

**Why it is here** — the OTHER hypothesis. It reaches no independent evidence line in this pipeline; it is here because it is among the ANL candidates most similar to a protein with MEASURED activity on the nylon-6,6 diad: 29.8% identity to AS23, a measured producer.

**Family (UniProt)** SIMILARITY: Belongs to the ATP-dependent AMP-binding enzyme family. .

## A1YAM3 — Amide synthetase

**Organism** Amycolatopsis orientalis (Nocardia orientalis)  
**Length** 521.0 aa  

**Why it is here** — the OTHER hypothesis. It reaches no independent evidence line in this pipeline; it is here because it is among the ANL candidates most similar to a protein with MEASURED activity on the nylon-6,6 diad: 30.6% identity to AS9, a measured producer.

**Family (UniProt)** SIMILARITY: Belongs to the ATP-dependent AMP-binding enzyme family. .

**Literature** [17136886](https://pubmed.ncbi.nlm.nih.gov/17136886/)

