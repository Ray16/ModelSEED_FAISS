# Candidates worth a bench slot

8 proteins, each supported by at least 3 independent observations that are not merely family membership. Ordered by how many lines agree, then by substrate distance.

**None of these has been assayed on nylon-6,6.** Convergence means several methods that fail differently happen to agree, which is a reason to spend a bench slot and not evidence that the enzyme works. With three measured positives in the reference panel, no ranking here can be validated.

---

## Q9RDS6 — Gamma-glutamylpolyamine synthetase GlnA2 (EC 6.3.1.-)

**Organism** Streptomyces coelicolor (strain ATCC BAA-471 / A3(2) / M145)  
**Gene** `glnA2 SCO2241`  
**Length** 453.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, independent_geometric_score

**Resolved**

- substrate distance **0.873** (own_curated_reaction)
- AlphaFold pLDDT **92.7**
- measured K_M **0.13 mM**
- curated ligand `CHEBI:29985`

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `cadaverine` — C5 diamine - ONE methylene short of HMDA
- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=; CATALYTIC ACTIVITY: Reaction=spermine + L-glut

**Function** FUNCTION: Involved in the catabolism of polyamines. Catalyzes the ATP-dependent gamma-glutamylation of polyamines. Substrates include putrescine, cadaverine, spermidine and spermine, with a preference for short-chain polyamine putrescine (PubMed:35409114). No complementation of the L-glutamine auxotrophy of an E.coli glnA mutant (PubMed:16932908). Together with GlnA3, enables survival of S.coelico

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

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

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=L-aspartate + L-methionine + ATP = L-aspartyl-L-methionine + ADP + phosphate + H(+); Xref=Rhea:RHEA:76447, ChEBI:CHEBI:15378, ChEBI:CHEBI:29991, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57844, ChEBI:CHEBI:195235, ChEBI:CHEBI:456216; Evidence=

**Function** FUNCTION: L-amino acid ligase, which preferentially catalyzes the formation of L-aspartyl-L-methionine dipeptide from L-aspartate and L-methionine in the presence of ATP. Less active with L-asparagine and L-methionine as substrates. Less active with L-aspartate and either L-phenylalanine, L-valine, L-leucine or L-isoleucine as substrates. Decreased activity when L-methionine is substituted with se

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

## I6Y0R5 — Dihydrofolate synthase/folylpolyglutamate synthase (DHFS / FPGS) (EC 6.3.2.12) (EC 6.3.2.17) (Folylpoly-gamma-glutamate 

**Organism** Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)  
**Gene** `folC Rv2447c`  
**Length** 487.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.803** (own_modelseed_reaction)
- its EC class reaches 0.382
- AlphaFold pLDDT **92.5**
- structure `2VOS` at 2.0 Å
- in-line adenylation geometry 6.12 Å
- curated ligand `CHEBI:17839`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=7,8-dihydropteroate + L-glutamate + ATP = 7,8-dihydrofolate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23584, ChEBI:CHEBI:15378, ChEBI:CHEBI:17839, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57451, ChEBI:CHEBI:456216; EC=6.3.2.12; Evidence=; CATALYTIC ACTIVITY: Reaction=(6S)-5,6,7,8-tetrahydrofolyl-(gamma-L-Glu)(n) + L-glutamate + ATP = (6S)-5,6

**Function** FUNCTION: Catalyzes the addition of a glutamate residue to dihydropteroate (7,8-dihydropteroate or H2Pte) to form dihydrofolate (7,8-dihydrofolate monoglutamate or H2Pte-Glu) (PubMed:23118010). Also catalyzes successive additions of L-glutamate to tetrahydrofolate, leading to folylpolyglutamate derivatives (By similarity). .; FUNCTION: Is involved in the bioactivation of the antituberculous drug p

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the folylpolyglutamate synthase family. .

**Note** MISCELLANEOUS: Mutations within this gene in the H2Pte binding pocket are responsible for PAS resistance in M.tuberculosis clinical isolates and laboratory strains. FolC-linked PAS resistance is mediated by altered substrate specificity that results in the failure to generate levels of H2PtePAS-Glu that are necessary to inhibit DHFR (DfrA); therefore, a blockage of PAS bioactivation causes PAS res

**BRENDA** `6.3.2.12;` — an entry exists; the API needs credentials this project does not have

**Literature** [9634230](https://pubmed.ncbi.nlm.nih.gov/9634230/), [12657046](https://pubmed.ncbi.nlm.nih.gov/12657046/), [16754987](https://pubmed.ncbi.nlm.nih.gov/16754987/), [21969609](https://pubmed.ncbi.nlm.nih.gov/21969609/), [23779105](https://pubmed.ncbi.nlm.nih.gov/23779105/), [23118010](https://pubmed.ncbi.nlm.nih.gov/23118010/), [24366731](https://pubmed.ncbi.nlm.nih.gov/24366731/), [18566510](https://pubmed.ncbi.nlm.nih.gov/18566510/)

## P08192 — Dihydrofolate synthase/folylpolyglutamate synthase (DHFS / FPGS) (EC 6.3.2.12) (EC 6.3.2.17) (Folylpoly-gamma-glutamate 

**Organism** Escherichia coli (strain K12)  
**Gene** `folC dedC b2315 JW2312`  
**Length** 422.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.803** (own_modelseed_reaction)
- its EC class reaches 0.382
- AlphaFold pLDDT **95.7**
- structure `1W78` at 1.82 Å
- in-line adenylation geometry 6.13 Å
- measured K_M **0.0006 mM**
- curated ligand `CHEBI:17839`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=7,8-dihydropteroate + L-glutamate + ATP = 7,8-dihydrofolate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23584, ChEBI:CHEBI:15378, ChEBI:CHEBI:17839, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57451, ChEBI:CHEBI:456216; EC=6.3.2.12; Evidence=; CATALYTIC ACTIVITY: Reaction=(6S)-5,6,7,8-tetrahydrofolyl-(gamma-L-Glu)(n) + L-glutamate + ATP = (6S)-5,6

**Function** FUNCTION: Functions in two distinct reactions of the de novo folate biosynthetic pathway. Catalyzes the addition of a glutamate residue to dihydropteroate (7,8-dihydropteroate or H2Pte) to form dihydrofolate (7,8-dihydrofolate monoglutamate or H2Pte-Glu). Also catalyzes successive additions of L-glutamate to tetrahydrofolate or 10-formyltetrahydrofolate or 5,10-methylenetetrahydrofolate, leading t

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the folylpolyglutamate synthase family. .

## Q81RQ8 — Citryl-spermidine/3,4-dihydroxybenzoyl-citryl-spermidine:spermidine ligase (EC 6.3.2.-) (Petrobactin biosynthesis protei

**Organism** Bacillus anthracis  
**Gene** `asbB GBAA_1982`  
**Length** 612.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.789** (own_curated_reaction)
- AlphaFold pLDDT **96.1**
- structure `3TO3` at 2.382 Å
- in-line adenylation geometry 5.2 Å

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=N(8)-citryl-spermidine + spermidine + ATP = N(8),N'(8)-citryl-bis(spermidine) + AMP + diphosphate + H(+); Xref=Rhea:RHEA:63812, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57834, ChEBI:CHEBI:149586, ChEBI:CHEBI:149592, ChEBI:CHEBI:456215; Evidence=; CATALYTIC ACTIVITY: Reaction=N(1)-(3,4-dihydroxybenzoyl)-N(8)-citryl-spermidine + spermidine + A

**Function** FUNCTION: Involved in the biosynthesis of petrobactin, a catecholate siderophore that functions in both iron acquisition and virulence (PubMed:16875672, PubMed:17189355, PubMed:22408253). Catalyzes the ATP-dependent condensation of spermidine with N(8)-citryl-spermidine or N(1)-(3,4-dihydroxbenzoyl)-N(8)-citryl-spermidine, two intermediates in petrobactin biosynthesis pathway (PubMed:18758617, Pub

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

## P9WJL3 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)  
**Gene** `murE Rv2158c MTCY270.10`  
**Length** 535.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **91.5**
- structure `2XJA` at 3.0 Å
- in-line adenylation geometry 6.2 Å
- crystallographic ligand `UAG`
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## G7CMG5 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Mycolicibacterium thermoresistibile (strain ATCC 19527 / DSM 44167 / CIP 105390 / JCM 6362  
**Gene** `murE KEK_21058`  
**Length** 506.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.764** (own_curated_reaction)
- its EC class reaches 0.764
- structure `8G6P` at 1.45 Å
- in-line adenylation geometry 6.44 Å
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

**Literature** [36944394](https://pubmed.ncbi.nlm.nih.gov/36944394/)

## Q46E79 — 3-methyl-D-ornithine--L-lysine ligase (EC 6.3.2.59) ((3R)-3-methyl-D-ornithine:L-lysine ligase) (Pyrrolysine biosynthesi

**Organism** Methanosarcina barkeri (strain Fusaro / DSM 804)  
**Gene** `pylC Mbar_A0837`  
**Length** 363.0 aa  

**Why it is here** — 3 independent lines: lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.667** (own_modelseed_reaction)
- its EC class reaches 0.667
- AlphaFold pLDDT **94.8**
- structure `4FFL` at 1.5 Å
- in-line adenylation geometry 6.61 Å
- crystallographic ligand `0TK`
- curated ligand `CHEBI:32551;CHEBI:57668`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(3R)-3-methyl-D-ornithine + L-lysine + ATP = (3R)-3-methyl-D-ornithyl-N(6)-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:32763, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:64642, ChEBI:CHEBI:64643, ChEBI:CHEBI:456216; EC=6.3.2.59; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:32764; Evidence=

**Function** FUNCTION: Is required for the biosynthesis of pyrrolysine (By similarity). Catalyzes the ATP-dependent ligation between (3R)-3-methyl-D-ornithine and L-lysine, leading to (3R)-3-methyl-D-ornithyl-N6-L-lysine (Probable). .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 magnesium ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the PylC family. .

