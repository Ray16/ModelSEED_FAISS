#!/usr/bin/env python
"""Side-by-side: ECFP4+DRFP (ecfp4_drfp) vs RXNFP."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # project rule: all figure text >= 18pt, dpi 300
FS = _style.FS

HERE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA=os.path.join(HERE,"data"); RESULTS=os.path.join(HERE,"results")
FIG=os.path.join(HERE,"figures","benchmark"); CACHE=os.path.join(HERE,"cache")
df=pd.read_parquet(os.path.join(DATA,"dataset.parquet")); ec1=df.ec1.to_numpy()
res=pd.read_csv(os.path.join(RESULTS,"results.csv")); knn=pd.read_csv(os.path.join(RESULTS,"knn_results.csv"))
A,B="ecfp4_drfp","rxnfp"
NAME={"ecfp4_drfp":"ECFP4+DRFP","rxnfp":"RXNFP"}
EC1_NAME={"1":"Oxidoreductases","2":"Transferases","3":"Hydrolases","4":"Lyases",
          "5":"Isomerases","6":"Ligases","7":"Translocases"}
EC1_COLOR={"1":"#0072B2","2":"#E69F00","3":"#009E73","4":"#CC79A7","5":"#56B4E9","6":"#D55E00","7":"#9a9a9a"}

def matched(e):
    s=res[(res.embedding==e)&(res.algo=="agglo_average_cos")&(res.params=="k=800")].iloc[0]; return s
def k15(e): return knn[(knn.embedding==e)&(knn.k==15)].iloc[0]

fig=plt.figure(figsize=(17,13))
gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=0.30,wspace=0.12)
# --- top: t-SNE maps ---
for col,e in enumerate([A,B]):
    ax=fig.add_subplot(gs[0,col]); Y=np.load(os.path.join(CACHE,"tsne_coords",e+".npy"))
    for cls in ["7","1","2","3","4","5","6"]:
        m=ec1==cls
        if m.any(): ax.scatter(Y[m,0],Y[m,1],s=3,c=EC1_COLOR[cls],alpha=0.55,linewidths=0,rasterized=True)
    kk=k15(e); ax.set_title(f"{NAME[e]}   (kNN EC3 {kk.ec3_acc:.2f})",fontsize=FS,pad=6)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]; ax.set_aspect("equal")
handles=[Line2D([0],[0],marker='o',ls='',ms=12,mfc=EC1_COLOR[c],mec='none',label=f"{c} {EC1_NAME[c]}") for c in "1234567" if (ec1==c).any()]
fig.legend(handles=handles,loc="lower center",ncol=7,frameon=False,fontsize=FS,bbox_to_anchor=(0.5,-0.04))
# --- bottom: metric bars ---
sa,sb=matched(A),matched(B); ka,kb=k15(A),k15(B)
metrics=[("EC3 AMI\n(cluster)",sa.ec3_ami,sb.ec3_ami),
         ("EC3 homog.\n(cluster)",sa.ec3_homogeneity,sb.ec3_homogeneity),
         ("EC3 purity\n(cluster)",sa.ec3_purity,sb.ec3_purity),
         ("full-EC AMI\n(cluster)",sa.ec_ami,sb.ec_ami),
         ("EC1 acc\n(15-NN)",ka.ec1_acc,kb.ec1_acc),
         ("EC3 acc\n(15-NN)",ka.ec3_acc,kb.ec3_acc),
         ("EC3 acc @conf\n(15-NN)",ka.ec3_acc_conf,kb.ec3_acc_conf)]
ax=fig.add_subplot(gs[1,:]); x=np.arange(len(metrics)); w=0.38
ax.bar(x-w/2,[m[1] for m in metrics],w,color="#0072B2",label=NAME[A])
ax.bar(x+w/2,[m[2] for m in metrics],w,color="#D55E00",label=NAME[B])
for i,m in enumerate(metrics):
    ax.text(i-w/2,m[1]+.014,f"{m[1]:.2f}",ha="center",fontsize=FS)
    ax.text(i+w/2,m[2]+.014,f"{m[2]:.2f}",ha="center",fontsize=FS)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics],fontsize=FS)
ax.set_ylim(0,1.12); ax.set_ylabel("score",fontsize=FS)
ax.tick_params(axis="y",labelsize=FS)
# legend in the empty upper band, anchored over the short full-EC-AMI column
ax.legend(frameon=False,fontsize=FS,loc="upper left",ncol=1,bbox_to_anchor=(0.40,1.02))
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
ax.set_title("Clustering (agglo-avg, k=800) and 15-NN retrieval metrics",fontsize=FS,pad=8)
fig.suptitle("ECFP4+DRFP vs RXNFP — reaction embedding comparison",fontsize=FS,y=1.0)
out=os.path.join(FIG,"fig_compare_ecfp4drfp_vs_rxnfp.png")
fig.savefig(out,dpi=300,bbox_inches="tight"); print("wrote",out)
