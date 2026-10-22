"""Regenerate the project status slides from saved evidence and the current run."""
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RESULTS=PROJECT/'results'
RUN=RESULTS/'boltz2_activation_20261016'
EXPERIMENT=Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase')
FIG=HERE/'status_figures'
FIG.mkdir(exist_ok=True)

def read_tsv(path):
    return list(csv.DictReader(path.open(),delimiter='\t'))

def esc(text):
    return str(text).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')

def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    master=json.loads((RESULTS/'master_candidates_summary.json').read_text())
    shortlist=json.loads((RESULTS/'shortlist_summary.json').read_text())
    convergence=json.loads((RESULTS/'converged_summary.json').read_text())
    kinetics=json.loads((RESULTS/'kinetics_summary.json').read_text())
    poses=defaultdict(list)
    for r in csv.DictReader((EXPERIMENT/'results/structural_proxies/tables/continuous_pose_geometry.csv').open()):
        if r['group']=='calibration' and r['id']!='SfaB':poses[r['id']].append((float(r['distance_A']),float(r['angle_deg'])))
    activity={r['id']:float(r['activity_N66L1_ugml_avg']) for r in csv.DictReader((EXPERIMENT/'assay_data/panel_activity.csv').open())}
    ids=sorted(poses,key=lambda i:int(i[2:]))
    A=np.array([sum(a/d for d,a in poses[i])/sum(1/d for d,a in poses[i]) for i in ids])
    yields=np.array([activity[i] for i in ids]);top3={'AS9','AS17','AS23'}
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9.2,4.3),layout='constrained')
    for label,color,marker,mask in [
        ('At or below SfaB mean','#9EA5AD','o',yields<=activity['SfaB']),
        ('Other above-SfaB means','#7B3294','o',np.array([i not in top3 for i in ids])&(yields>activity['SfaB'])),
        ('Three strongest producers','#0072B2','o',np.array([i in top3 for i in ids]))]:
        ax.scatter(A[mask],np.maximum(yields[mask],.001),c=color,marker=marker,s=48,label=label,edgecolor='white',linewidth=.5,zorder=3)
    ax.axhline(activity['SfaB'],color='#D55E00',ls='--',lw=1.1,label='Pooled SfaB mean')
    for i in top3:
        j=ids.index(i)
        offset={'AS9':(6,3),'AS17':(8,10),'AS23':(-35,-8)}[i]
        ax.annotate(i,(A[j],yields[j]),xytext=offset,textcoords='offset points',fontsize=12,fontweight='bold')
    ax.set(yscale='log',xlabel=r'Inverse-distance-weighted attack angle $A$ (degrees)',ylabel=r'Mean diad concentration ($\mu$g/mL)')
    ax.grid(axis='y',alpha=.18);ax.legend(loc='upper left',fontsize=9,frameon=False)
    fig.savefig(FIG/'proxy_activity.pdf');plt.close(fig)
    route=master['by_route']
    labels=['clade','domain','enzyme_name','precomputed_link','active_homolog','reaction','kegg_ortholog','assayed','proxied']
    fig,ax=plt.subplots(figsize=(8.4,4.3),layout='constrained')
    values=[route[k] for k in labels]
    ax.barh(labels[::-1],values[::-1],color=['#0072B2']*len(labels))
    ax.set_xscale('log');ax.set_xlabel('Candidates assigned to each discovery route (log scale)')
    for y,v in enumerate(values[::-1]):ax.text(v*1.12,y,f'{v:,}',va='center',fontsize=11)
    ax.set_xlim(8,85000);ax.grid(axis='x',alpha=.12);fig.savefig(FIG/'routes.pdf');plt.close(fig)
    proxy_rows=[]
    for i in sorted(top3,key=lambda i:-A[ids.index(i)]):
        rank=1+int((A>A[ids.index(i)]).sum())
        proxy_rows.append(f'{i} & {rank} & {A[ids.index(i)]:.1f} & {activity[i]:.2f} '+r'\\')
    cols=['mechanism_family','accession','mg_count','phosphate']
    # Generated status is a snapshot of filesystem evidence, never a future claim.
    done=list((RUN/'state').glob('*.done.json'))
    fails=list((RUN/'state').glob('*.failure.json'))
    pdbs=sum(1 for _ in (RUN/'outputs').glob('*/by_seed/*/s42/*_model_*.pdb'))
    checked=json.loads((RUN/'evidence/startup_verification.json').read_text()) if (RUN/'evidence/startup_verification.json').exists() else {}
    resident=len(checked.get('resident_gpu_processes',[]))
    coord=json.loads((RUN/'state/coordinator.json').read_text())
    status='Complete' if coord.get('status')=='complete' else 'Running' if resident else 'Workers launched'
    entries=defaultdict(list)
    for item in manifest['inputs']:entries[item['mechanism_family']].append(item)
    completed={json.loads(p.read_text())['accession']:json.loads(p.read_text()) for p in done}
    names={'Q2FWC5':'LdmS','P9WJL3':'MurE (Mtb)','Q2FZP6':'MurE (Sa)','Q9WY79':'MurE (Tm)',
           'P22188':'MurE (Ec)','P0AES0':'Gss (Ec)','P90518':'GSP (Cf)'}
    ordered=sorted([i for i in manifest['inputs'] if i['accession'] in completed],
                   key=lambda i:-completed[i['accession']]['A_deg'])
    from matplotlib.patches import Patch
    for page,subset in enumerate([ordered[:9],ordered[9:]],1):
        fig,(ax,dx)=plt.subplots(1,2,figsize=(11,4.6),layout='constrained',sharey=True,gridspec_kw={'width_ratios':[1.8,1]})
        for j,item in enumerate(subset):
            value=completed.get(item['accession']);color='#0072B2' if item['phosphate']=='PA' else '#D55E00'
            if value:
                ax.barh(j,value['A_deg'],color=color,height=.65)
                ax.text(value['A_deg']+2,j,f'{value["A_deg"]:.1f}',va='center',fontsize=13)
                dx.barh(j,value['Dh_A'],color=color,height=.65)
                dx.text(value['Dh_A']+.3,j,f'{value["Dh_A"]:.2f}',va='center',fontsize=13)
            else:
                ax.text(3,j,'pending',va='center',color='#999999',fontsize=13)
                dx.text(.3,j,'pending',va='center',color='#999999',fontsize=13)
        ylabels=[names.get(i['accession'],i['gene'].split()[0])+ ' / ' +i['accession'] for i in subset]
        ax.set_yticks(range(len(subset)),ylabels,fontsize=13);ax.set_ylim(len(subset)-.4,-.6);dx.tick_params(axis='y',left=False,labelleft=False)
        ax.set_xlim(0,200);dx.set_xlim(0,max([v['Dh_A'] for v in completed.values()]+[20])*1.3)
        ax.set_xlabel(r'Weighted angle $A$ (degrees)',fontsize=13);dx.set_xlabel(r'Harmonic distance $D_h$ ($\AA$)',fontsize=13)
        for panel in [ax,dx]:panel.grid(axis='x',alpha=.15);panel.set_axisbelow(True);panel.tick_params(axis='x',labelsize=12)
        ax.legend(handles=[Patch(facecolor='#0072B2',label=r'$P_\alpha$'),Patch(facecolor='#D55E00',label=r'$P_\gamma$')],loc='lower right',fontsize=12,frameon=False)
        fig.savefig(FIG/f'new_candidate_geometry_{page}.pdf');plt.close(fig)
    family_names={'NIS':'NIS / IucA--IucC','Mur':'MurE ligases','PylC':'PylC (ATP-grasp)','LdmS':'LdmS (ATP-grasp)','GS':'GlnA2 / PuuA','GspS':'Glutathionylspermidine synthetases','DdaG':'DdaG adenylating ligase'}
    metal_rows=[]
    for key in ['NIS','DdaG','Mur','PylC','LdmS','GS','GspS']:
        group=entries[key];first=group[0]
        phosphate=r'$P_\alpha$' if first['phosphate']=='PA' else r'$P_\gamma$'
        metal_rows.append(f'{family_names[key]} & {len(group)} & {first["mg_count"]} & {phosphate} '+r'\\')
    provenance={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'source_tables':str(RESULTS),
                'run_manifest':str(RUN/'manifest.json'),'outputs_written':pdbs,'complete_qc_ensembles':len(done),
                'historical_failure_records':len(fails),'run_status':coord.get('status'),'verified_gpu_processes_at_startup':resident,
                'proxy_source':str(EXPERIMENT/'results/structural_proxies/attempts/proxy_performance_20261011/results.json'),
                'caveat':'Geometry figures use the historical 48-homolog assay panel; new candidate results are not substituted until all output/QC checks pass.'}
    (HERE/'status_sources.json').write_text(json.dumps(provenance,indent=2))
    text=r'''% Project status update generated by build_status_deck.py from saved evidence.
\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{default}
\usecolortheme{seahorse}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}
\usepackage{graphicx,booktabs,xcolor,tikz}
\usepackage[skins]{tcolorbox}
\usetikzlibrary{positioning,arrows.meta}
\graphicspath{{status_figures/}{../figures/}}
\definecolor{cAccent}{HTML}{0072B2}
\definecolor{cGood}{HTML}{1E8449}
\definecolor{cOrange}{HTML}{D55E00}
\setbeamercolor{frametitle}{fg=structure!65!black,bg=structure!28!white}
\setbeamertemplate{frametitle}{\nointerlineskip\begin{beamercolorbox}[wd=\paperwidth,ht=3.0ex,dp=1.4ex,leftskip=0.35cm]{frametitle}\usebeamerfont{frametitle}\insertframetitle\end{beamercolorbox}}
\setbeamerfont{frametitle}{size=\large,series=\bfseries}
\setbeamercolor{block title}{bg=structure!20!white,fg=black}
\setbeamercolor{block body}{bg=structure!8!white}
\setbeamertemplate{footline}[frame number]
\newcommand{\takeaway}[1]{\vfill\begin{tcolorbox}[colback=structure!10!white,colframe=structure!55!black,boxrule=.4pt,left=5pt,right=5pt,top=3pt,bottom=3pt]\small #1\end{tcolorbox}}
\newcommand{\good}[1]{{\color{cGood}\textbf{#1}}}
\newcommand{\source}[1]{\par\smallskip{\tiny\color{gray}#1}}
\title{\textbf{Finding enzymes for nylon-6,6 amide synthesis}}
\subtitle{Reaction retrieval, evidence-based candidates, and structural triage}
\date{16 October 2026}
\begin{document}
\begin{frame}[plain]
\titlepage
\begin{center}\large\textbf{17 candidates \quad $\times$ \quad 50 Boltz-2 poses}\par\medskip
\normalsize Protein + ATP + adipate$^{2-}$ + family-specific Mg$^{2+}$\end{center}
\end{frame}

\begin{frame}{Target: activate a C6 diacid, then form an amide}
\begin{center}
\begin{tikzpicture}[node distance=8mm,box/.style={draw=structure!60,fill=structure!8,rounded corners,align=center,text width=3.5cm,minimum height=1.05cm},>=Stealth]
\node[box] (acid) {\textbf{Adipate$^{2-}$}\\C6 diacid};
\node[box,right=of acid] (activation) {\textbf{ATP-dependent activation}\\enzyme + Mg$^{2+}$};
\node[box,right=of activation] (amide) {\textbf{Amide bond formation}\\HMDA supplies the amine};
\draw[->,thick] (acid)--(activation);\draw[->,thick] (activation)--(amide);
\end{tikzpicture}
\end{center}
\medskip
\begin{itemize}
\item Discovery asks which enzymes could accept the two monomers.
\item The current cofold batch tests \textbf{first-step activation geometry}.
\item HMDA acceptance and repeated chain extension remain separate questions.
\end{itemize}
\takeaway{A favorable acid--ATP geometry is a reason to prioritize an assay; it does not establish the full nylon-6,6 reaction.}
\end{frame}

\begin{frame}{Saved candidate sets answer different questions}
\centering
\begin{tabular}{lrp{7.4cm}}
\toprule\textbf{Set} & \textbf{Proteins} & \textbf{Meaning}\\\midrule
Search pool & \textbf{55,125} & Includes deep family and clade expansion.\\[4pt]
Practical shortlist & \textbf{317} & Bench eligibility with diversity caps.\\[4pt]
Uncapped evidence set & \textbf{84} & At least two discriminating evidence lines and bench eligibility.\\[4pt]
Converged set & \good{17} & At least three discriminating evidence lines; this cofold batch.\\[4pt]
Proposed first assay round & \textbf{11} & Three hypotheses; experimental plan, not completed assays.\\\bottomrule
\end{tabular}
\takeaway{The 317-protein shortlist and the uncapped 84-protein evidence set are different selections: 44 of the 84 are outside the diversity-capped shortlist.}
\source{Saved TSV/JSON outputs, checked 16 Oct 2026; QUALIFIED.tsv; converged\_candidates.tsv; ASSAY\_ROUND.md.}
\end{frame}

\begin{frame}{Reaction similarity retrieves; chemistry decides relevance}
\begin{columns}[T]
\begin{column}{.48\textwidth}
\begin{block}{Whole-reaction fingerprints}
ECFP4-diff + DRFP\par\medskip
ATP and CoA changes can dominate cosine similarity.\par\medskip
An acyl-CoA query can retrieve \textbf{thioester hydrolases} that form no amide.
\end{block}
\end{column}
\begin{column}{.48\textwidth}
\begin{block}{Independent chemistry screen}
Reconstruct the free amine by cutting product amide bonds.\par\medskip
Check the reacting nitrogen, substrate size, and reaction balance.\par\medskip
\textbf{726} free-amine acylations; \textbf{480} eligible reactions.
\end{block}
\end{column}
\end{columns}
\takeaway{Retrieval no longer requires a high cosine score. Substrate resemblance is an explicit chemical heuristic, not a fitted activity predictor.}
\source{ModelSEED master b539adb6; screen\_summary.json; 564 offline regression tests pass.}
\end{frame}

\begin{frame}{Nine recorded routes produce the 55,125-protein pool}
\centering\includegraphics[height=.65\textheight]{routes.pdf}
\takeaway{Most rows are homologs. The \texttt{active\_homolog} route adds \textbf{822} relatives of measured producers, including proteins whose annotated chemistry is a thioester reaction.}
\source{master\_candidates\_summary.json. Route counts describe merged-row attribution, not independent biological hypotheses.}
\end{frame}

\begin{frame}{Evidence counts were corrected for shared information}
\begin{columns}[T]
\begin{column}{.48\textwidth}
\begin{block}{Retained discriminating observations}
\begin{itemize}
\item Free-diamine acylation
\item Own-reaction substrate chemistry
\item Distinct ligand evidence
\item Cofactor-bound structure
\item Available external geometry analysis
\end{itemize}
\end{block}
\end{column}
\begin{column}{.48\textwidth}
\begin{block}{Avoid counting the same fact twice}
Family membership is excluded from the headline count.\par\medskip
A curated binding ligand that merely restates the curated reaction is collapsed.\par\medskip
Headline set: \textbf{28 $\rightarrow$ 17}.
\end{block}
\end{column}
\end{columns}
\takeaway{These are evidence-availability counts, not calibrated success probabilities. The external geometry axis currently counts score availability, including unfavorable scores.}
\source{converged.py; converged\_summary.json. The 17 span 14 gene symbols and nine recorded fold signatures.}
\end{frame}

\begin{frame}{Representative candidates in the 17-protein set}
\centering\small
\begin{tabular}{llcp{6.1cm}}
\toprule\textbf{Enzyme} & \textbf{Accession} & \textbf{Lines} & \textbf{Why examine it structurally?}\\\midrule
AsbB & Q81RQ8 & 4 & Free-polyamine acylation; cofactor structure.\\[4pt]
MurE & P9WJL3 & 4 & Diamine-like native acceptor; bulky native acid scaffold.\\[4pt]
PylC & Q46E79 & 4 & ATP-grasp ligation; two-Mg annotation.\\[4pt]
GlnA2 & Q9RDS6 & 3 & Native polyamine repertoire includes cadaverine.\\[4pt]
GspSA & P0AES0 & 3 & ATP-dependent polyamine ligation; structural and kinetic evidence.\\\bottomrule
\end{tabular}\par
\medskip
All \textbf{17} are retained for the new cofolds, including candidates that may fail geometrically.
\takeaway{Correction to the older deck: GlnA2 now has \textbf{three} counted lines. Its saved $K_M=0.13$ mM is for \textbf{ATP}, not cadaverine.}
\source{converged\_candidates.tsv; curated\_properties.tsv. No new nylon-6,6 activity is claimed.}
\end{frame}

\begin{frame}{The assay plan preserves three competing hypotheses}
\centering\small
\begin{tabular}{lp{5.7cm}p{3.6cm}}
\toprule\textbf{Hypothesis} & \textbf{Proposed proteins} & \textbf{Evidence and gap}\\\midrule
Chemistry (4) & GlnA2, AsbB, MurE, PylC & Mechanistic plausibility; no target assay.\\[7pt]
Producer homology (4) & A0A7Y0L7E5, G8U0Y4, A0A3E0HG69, R4SVL2 & 50--65\% identity to a producer; target function untested.\\[7pt]
Acceptor chemistry (3) & P0AES0, P21340, Q9HL57 & Measured polyamine chemistry; the latter two use CoA donors.\\\bottomrule
\end{tabular}
\takeaway{The \textbf{11-protein assay plan} is distinct from the \textbf{17-protein cofold set}. AS9 or AS17 is proposed as an additional assay positive control.}
\source{ASSAY\_ROUND.md; active\_homolog\_summary.json. No protein orders or new assays have been performed here.}
\end{frame}

\begin{frame}{The proxy is the inverse-distance-weighted mean angle}
\begin{center}
\Large\[
 A=\frac{\displaystyle\sum_{i=1}^{50}\theta_i/d_i}{\displaystyle\sum_{i=1}^{50}1/d_i}
 \qquad
 D_h=\frac{50}{\displaystyle\sum_{i=1}^{50}1/d_i}
\]
\end{center}
\begin{itemize}
\item $d_i$: nearest donor carboxylate oxygen to the reactive ATP phosphorus.
\item $\theta_i$: O$_{\rm donor}$--P--O$_{\rm leaving}$ angle; larger is more in-line.
\item All 50 poses contribute; closer poses carry greater weight.
\end{itemize}
\takeaway{$A$ has units of degrees and is unchanged by uniformly scaling all distances. Report \textbf{$D_h$ and distance support} alongside it. No hard 4 \AA\ gate is used in $A$.}
\source{Existing docking\_angle\_review.py; exact formula independently reproduced on the saved historical panel.}
\end{frame}

\begin{frame}{Historical reference panel: retrospective proxy enrichment}
\begin{columns}[T]
\begin{column}{.56\textwidth}
\small\begin{tabular}{lrrr}
\toprule\textbf{Producer} & \textbf{Rank} & \textbf{$A$ (deg)} & \textbf{$\mu$g/mL}\\\midrule
@@PROXY_ROWS@@
\bottomrule\end{tabular}
\medskip
\begin{itemize}
\item All three in the \textbf{top ten} of 48 homologs.
\item Top-three AUROC: \good{0.941}.
\item Seven above-SfaB means: AUROC \textbf{0.707}; \textbf{4/7} recovered at ten.
\end{itemize}
\end{column}
\begin{column}{.41\textwidth}
\begin{block}{Scope of the result}
Previously inspected panel; three standout producers.\par\medskip
Supports candidate shortlisting.\par\medskip
Does not establish exact yield ranking or transfer across enzyme families.
\end{block}
\end{column}
\end{columns}
\takeaway{The historical tuned smooth score $S$ is a different formula. Its AUROC of 0.978 must not be attributed to the requested $1/d$-weighted angle $A$.}
\source{simple\_geometry\_review\_20261010; proxy\_performance\_20261011. SfaB excluded from the 48-homolog ranking.}
\end{frame}

\begin{frame}{New candidate geometry (1/2): @@DONE@@ of 17 complete}
\centering\includegraphics[width=.98\linewidth,height=.62\textheight,keepaspectratio]{new_candidate_geometry_1.pdf}
\takeaway{This campaign's candidates, ordered by $A$ among completed ensembles. $D_h$ reports absolute proximity; colors distinguish the reactive ATP phosphate. No new activity data are available.}
\source{Boltz-2 exact, 50 poses per completed input; ATP + adipate$^{2-}$ + family-specific Mg. Snapshot: @@SNAPSHOT@@ UTC.}
\end{frame}
\begin{frame}{New candidate geometry (2/2): completed candidates only}
\centering\includegraphics[width=.98\linewidth,height=.62\textheight,keepaspectratio]{new_candidate_geometry_2.pdf}
\takeaway{Only completed 50-pose ensembles are plotted. Unfinished candidates are omitted. A low angle or large $D_h$ indicates weak modeled activation geometry; sampling and enzyme assembly still require interpretation.}
\source{Boltz-2 exact, 50 poses per completed input; ATP + adipate$^{2-}$ + family-specific Mg. Snapshot: @@SNAPSHOT@@ UTC.}
\end{frame}

\begin{frame}{Cofold the activation step with the correct ATP coordinate}
\centering\begin{tabular}{lcc}
\toprule\textbf{Native activation chemistry} & \textbf{Reactive P} & \textbf{Leaving axis}\\\midrule
ATP $\rightarrow$ AMP + PPi & $P_\alpha$ & P$_\alpha$--O3A\\[7pt]
ATP $\rightarrow$ ADP + P$_i$ & $P_\gamma$ & P$_\gamma$--O3B\\\bottomrule
\end{tabular}
\medskip
\begin{block}{Common input and sampling protocol}
One protein chain + ATP + \textbf{adipate$^{2-}$} + assigned Mg$^{2+}$.\par\smallskip
Boltz-2 \textbf{exact} mode; sequence-specific MSA; 50 diffusion samples; no pocket/attack restraints.
\end{block}
\takeaway{All 17 use ATP. The nucleotide product in the candidate's annotated reaction determines which phosphorus is measured. Mg stoichiometry is checked separately against family and paper evidence.}
\source{proxy\_policy.mechanism\_from\_reaction; charge verified with RDKit (C6H8O4, $-2$).}
\end{frame}

\begin{frame}{Per-active-site Mg assignments: seven one-Mg, ten two-Mg}
\centering\small
\begin{tabular}{lrrc}
\toprule\textbf{Catalytic family / subgroup} & \textbf{Proteins} & \textbf{Mg per input} & \textbf{Reactive P}\\\midrule
@@METAL_ROWS@@
\bottomrule\end{tabular}
\takeaway{Use the existing family/mechanism resolver; keep UniProt tallies as a cross-check. NIS occupancy can vary: the one-Mg assignment uses AsbB evidence and remains a modeling hypothesis for its relatives.}
\source{metal\_assignments.tsv; family\_cofactor\_queries.json; resolve\_handoff.py. Sources and confidence are saved per candidate.}
\end{frame}

\begin{frame}{A surrogate ion can make the deposited Mg count misleading}
\begin{columns}[T]
\begin{column}{.46\textwidth}
\begin{block}{LdmS: crystallographic observation}
7R8P / 7R8Q:\par\medskip
\Large one Mg$^{2+}$ + Na$^+$\normalsize\par\medskip
The second catalytic metal site is occupied by sodium under the crystallization conditions.
\end{block}
\end{column}
\begin{column}{.49\textwidth}
\begin{block}{LdmS: catalytic description}
Primary paper:\par\medskip
\Large two Mg$^{2+}$\normalsize\par\medskip
Two-metal-assisted phosphoryl transfer is the modeled state. Do not copy the surrogate-ion tally into the cofold.
\end{block}
\end{column}
\end{columns}
\takeaway{The paper's mechanistic description resolves this discrepancy: \textbf{Q2FWC5 is modeled with two Mg$^{2+}$ ions}.}
\source{Pederick et al., J. Biol. Chem. 298, 102392 (2022), doi:10.1016/j.jbc.2022.102392; PMCID PMC9486568.}
\end{frame}

\begin{frame}{Cofolding status: @@RUN_STATUS@@}
\begin{center}\Large\textbf{17 candidates $\times$ 50 poses = 850 requested structures}\end{center}
\medskip
\centering\begin{tabular}{lr}
\toprule\textbf{Milestone} & \textbf{Verified state}\\\midrule
Candidate inputs and chemistry checks & 17 prepared\\
Sequence-specific MSAs & 17 complete\\
Exclusive GPU workers at startup & 4 H100s on lambda13\\
Structures written at this snapshot & @@PDBS@@ / 850\\
Complete, assembly-checked ensembles & @@DONE@@ / 17\\\bottomrule
\end{tabular}
\takeaway{Every worker runs through \texttt{gpu\_reserve run}: one worker per GPU, disjoint CPU sets, explicit native thread limits, and resumable per-input claims.}
\source{Snapshot: @@SNAPSHOT@@ UTC. Manifest, worker logs, and startup verification saved with the campaign.}
\end{frame}

\begin{frame}{What happens after the structures are written}
\begin{center}
\begin{tikzpicture}[node distance=6mm,box/.style={draw=structure!60,fill=structure!8,rounded corners,align=center,text width=3.6cm,minimum height=1.0cm},>=Stealth]
\node[box] (qc) {\textbf{Output + assembly checks}\\50 models, sequence, ATP, acid, Mg};
\node[box,right=of qc] (geometry) {\textbf{Geometry ranking}\\$A$, $D_h$, and distance support};
\node[box,right=of geometry] (assay) {\textbf{Experimental triage}\\acid activation + HMDA acceptance};
\draw[->,thick](qc)--(geometry);\draw[->,thick](geometry)--(assay);
\end{tikzpicture}
\end{center}
\medskip\begin{itemize}
\item Keep all candidates; report missing or unfavorable geometry explicitly.
\item A low score can reflect poor ligand placement or an incomplete modeled assembly.
\item Check activation and amine acceptance before claiming diad synthesis.
\end{itemize}
\takeaway{The immediate deliverable is an auditable structural comparison of the \textbf{17 saved candidates}. Experimental activity remains the deciding evidence.}
\end{frame}

\begin{frame}{Files to use for the next review}
\small\begin{tabular}{lp{7.2cm}}
\toprule\textbf{Artifact} & \textbf{Purpose}\\\midrule
\texttt{converged\_candidates.tsv} & Saved 17-protein candidate set.\\[5pt]
\texttt{ASSAY\_ROUND.md} & Eleven proposed proteins across three hypotheses.\\[5pt]
\texttt{boltz2\_activation\_20261016/} & Current inputs, MSAs, predictions, and run records.\\[5pt]
\texttt{manifest.json} & Exact sequences, charges, Mg assignments, atom coordinates, and inference settings.\\[5pt]
\texttt{geometry\_ranking.tsv} & Populated from completed, assembly-checked ensembles.\\\bottomrule
\end{tabular}
\takeaway{Older cofold ensembles are preserved separately. This charge-explicit batch is not silently pooled with historical runs.}
\end{frame}
\appendix
\begin{frame}{Validation appendix: the previously assayed AS panel}
\centering\includegraphics[width=.94\linewidth,height=.70\textheight,keepaspectratio]{proxy_activity.pdf}
\takeaway{These AS proteins are \textbf{historical assay references}, not members of the new 17-candidate cofold batch. Means below 0.001 $\mu$g/mL use a plotting floor. Above-SfaB mean is a descriptive endpoint.}
\end{frame}
\end{document}
'''
    substitutions={'PROXY_ROWS':'\n'.join(proxy_rows),'METAL_ROWS':'\n'.join(metal_rows),'RUN_STATUS':status,
                   'PDBS':str(pdbs),'DONE':str(len(done)), 'SNAPSHOT':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}
    for k,v in substitutions.items():text=text.replace('@@'+k+'@@',v)
    assert '@@' not in text
    (HERE/'main.tex').write_text(text)
    print('Updated status deck:',len(manifest['inputs']),'inputs;',pdbs,'PDBs;',len(done),'QC-complete ensembles')

if __name__=='__main__':main()
