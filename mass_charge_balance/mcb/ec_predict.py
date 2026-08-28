"""
Predicted-EC support — kept deliberately SEPARATE from curated EC.

ModelSEED's own `ec_numbers` is curated and is what the main pipeline trusts. When it is
absent, an EC can be *predicted* from the reaction structure. Predicted EC is NOT validated,
so it is always tagged by source and its corrections land in their own `review-predicted`
tier — never auto-apply, never mixed into the curated-EC deliverable.

The loader is predictor-agnostic: it reads every `inputs/predicted_ec*.tsv` (rxnfp today;
CARE / an LLM predictor drop in as `inputs/predicted_ec_<name>.tsv` later). A prediction is
usable only if the predictor marked it confident AND in-domain (not OOD). When two predictors
agree on EC3 for the same reaction, that agreement is recorded (a stronger, but still
predicted, signal).

Columns expected per source file: rxn_id, pred_ec1, pred_ec2, pred_ec3, conf_ec3,
confident (True/False), ood (True/False); optional top1_ref_id, top1_ref_ec, metric_agrees.
"""
import glob
import os

import pandas as pd

from mcb.paths import INPUTS


def _source_name(path):
    b = os.path.basename(path)[len("predicted_ec"):].lstrip("_").rsplit(".", 1)[0]
    return b or "rxnfp"


def load_predicted_ec(min_conf=0.6):
    """rid -> dict(pred_ec1/2/3, conf, sources[list], usable[bool], analogue).

    `usable` = at least one source is confident+in-domain with conf_ec3 >= min_conf.
    If sources disagree on EC3, the highest-confidence usable call wins and `agree` is False.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(str(INPUTS), "predicted_ec*.tsv"))):
        src = _source_name(path)
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for _, r in df.iterrows():
            try:
                conf = float(r.get("conf_ec3", "") or 0)
            except ValueError:
                conf = 0.0
            usable = (r.get("confident", "") == "True" and r.get("ood", "") != "True"
                      and conf >= min_conf)
            rec = {"pred_ec1": r.get("pred_ec1", ""), "pred_ec2": r.get("pred_ec2", ""),
                   "pred_ec3": r.get("pred_ec3", ""), "conf": conf, "source": src,
                   "usable": usable, "analogue": r.get("top1_ref_id", ""),
                   "analogue_ec": r.get("top1_ref_ec", "")}
            prev = out.get(r["rxn_id"])
            if prev is None:
                rec["sources"] = [src]
                rec["by_source"] = {src: rec["pred_ec3"]}      # per-source EC3 for cross-checks
                rec["agree"] = True
                out[r["rxn_id"]] = rec
            else:
                prev["sources"].append(src)
                prev["by_source"][src] = rec["pred_ec3"]
                prev["agree"] = prev["agree"] and (prev["pred_ec3"] == rec["pred_ec3"])
                # prefer the usable, higher-confidence call as the representative
                if (rec["usable"], rec["conf"]) > (prev["usable"], prev["conf"]):
                    carry = {k: prev[k] for k in ("sources", "by_source", "agree")}
                    out[r["rxn_id"]] = rec
                    rec.update(carry)
    return out


def effective_ec(rid, curated_ec, preds, primary="rxnfp"):
    """Return (ec_string, source, confidence, corroboration).

    source is 'curated' (trusted), 'predicted:<name>' (tagged, review-only), or 'none'.
    When a `primary` source (default rxnfp) has a usable call it DRIVES the returned EC —
    the fix stays deterministic and attributable — and `corroboration` reports whether the
    other predictors agree at EC3:
        {'agree': bool|None, 'by_source': {name: ec3}, 'n_sources': int}
    Predicted EC is a full EC3 prefix so the same EC_CORROBORATION / ec_compatible gates
    apply, but the caller must route the result to a review-predicted tier.
    """
    if curated_ec not in ("", "null"):
        return curated_ec, "curated", 1.0, {"agree": None, "by_source": {}, "n_sources": 0}
    p = preds.get(rid)
    if not (p and p["usable"]):
        return "", "none", 0.0, {"agree": None, "by_source": {}, "n_sources": 0}
    by = p.get("by_source", {p["source"]: p["pred_ec3"]})
    # drive from the primary predictor when it has this reaction; else the representative
    driver_ec = by.get(primary, p["pred_ec3"])
    driver_src = primary if primary in by else p["source"]
    others = {s: e for s, e in by.items() if s != driver_src}
    agree = (all(e == driver_ec for e in others.values()) if others else None)
    tag = f"predicted:{driver_src}" + ("+" + "+".join(sorted(others)) if agree else "")
    return driver_ec, tag, p["conf"], {"agree": agree, "by_source": by, "n_sources": len(by)}
