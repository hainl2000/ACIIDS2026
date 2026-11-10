"""Updated versions of dataset1 / dataset2: same lncRNAs, diseases and miRNAs, associations refreshed from the newer
releases of the databases each benchmark was built from (paper, Sec. Datasets).

  lncRNA-disease : Lnc2Cancer 3.0 (all records, experimental by design) + LncRNADisease (human, experimental only:
                   records whose method field is a prediction tool are dropped). GeneRIF is not used.
  lncRNA-miRNA   : ENCORI hg38 miRNA-lncRNA (successor of starBase v2.0), sites with clipExpNum >= 1.
  miRNA-disease  : HMDD v4.0.
New matrices = old associations OR associations found in the new releases, restricted to the existing entities.
Every edge of the result is listed in provenance/<relation>_edges.csv (in_old, in_new, sources), so the new-only
edges can be used as a temporal validation set.

Mapping
  lncRNA  : HGNC canonical symbol (current symbol / unique previous symbol / unique alias), else the upper-cased name.
  disease : disease name -> DOID with the curated tables of NKHTrẻ2026/Dataset (DatasetCuration/disease_doid_mapping.csv
            for Lnc2Cancer / LncRNADisease, HMDD/alldata_v4_doid.csv for HMDD); only DOIDs of the dataset are kept.
  miRNA   : precursor (hsa-mir-181b-2) or family (mir181b, let7a) keys; a mature ENCORI name (hsa-miR-181b-5p) maps to
            every precursor of its family present in the dataset, hsa-miR-181a-2-3p only to hsa-mir-181a-2.

Usage: python build_updated_datasets.py            (writes our_dataset/<dataset>_v2026/, never touches the originals)
"""
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(REPO), "NKHTrẻ2026", "Dataset")
SRC = {
    "lnc2cancer": f"{DATA}/lnc2Cancer3/lncRNA.csv",
    "lncrnadisease": f"{DATA}/lncRNADiseaseDatabase/lncRNA.csv",
    "encori": f"{DATA}/encori/lncRNA/hg38_miRlncRNA.txt",
    "hmdd": f"{DATA}/HMDD/alldata_v4_doid.csv",
    "disease_doid": f"{DATA}/DatasetCuration/disease_doid_mapping.csv",
    "hgnc": f"{DATA}/HGNC/hgnc_complete_set.txt",
}
SRC["doid_obo"] = f"{DATA}/doid.obo"
PREDICTION_TOOLS = {"TAM", "LDAP", "LRLSLDA-LNCSIM1", "LRLSLDA-LNCSIM2", "RWRLNCD"}
MIN_CLIP = 1
TAG = "v2026"
# Disease-annotation convention of each original benchmark, measured on its own matrices: for an edge (x, d), the share
# of d's in-dataset Disease Ontology ancestors that are linked to x as well. dataset1 (Fu et al.): 83% (lnc-disease)
# / 86% (miRNA-disease) -> associations were propagated to ancestor diseases (true-path rule); dataset2 (Zhou et al.):
# 16% / 20% -> direct annotations. New associations follow the same convention.
PROPAGATE_TO_ANCESTORS = {"dataset1": True, "dataset2": False}
# Only NEWER releases of the databases a benchmark was built from (paper, Sec. Datasets) may add edges.
#   dataset1 (paper "Dataset 2", Fu et al. 2018): LncRNADisease (2012) -> v3.0, Lnc2Cancer (2016) -> 3.0, GeneRIF (not
#            used), starBase v2.0 -> ENCORI, HMDD v2.0 -> v4.0.
#   dataset2 (paper "Dataset 1", Zhou et al. 2022): LncRNADisease v2.0 -> v3.0, starBase v2.0 -> ENCORI, HMDD v3.0 -> v4.0.
#            Lnc2Cancer is excluded: the downloaded release is 3.0, the same release the benchmark already used.
LNC_DISEASE_SOURCES = {"dataset1": {"Lnc2Cancer3.0", "LncRNADisease"}, "dataset2": {"LncRNADisease"}}


# ------------------------------------------------------------------ name normalisation
def load_hgnc():
    h = pd.read_csv(SRC["hgnc"], sep="\t", dtype=str, low_memory=False).fillna("")
    cur, prev, alias = set(), defaultdict(set), defaultdict(set)
    for r in h.itertuples(index=False):
        s = r.symbol.upper()
        cur.add(s)
        for p in r.prev_symbol.strip('"').split("|"):
            if p:
                prev[p.upper()].add(s)
        for a in r.alias_symbol.strip('"').split("|"):
            if a:
                alias[a.upper()].add(s)

    def canon(name):
        k = str(name).strip().upper()
        if k in cur:
            return k
        for table in (prev, alias):
            if len(table.get(k, ())) == 1:
                return next(iter(table[k]))
        return k
    return canon


MIR = re.compile(r"^(?:hsa-)?(mir|let)-?(\d+[a-z]*)(?:-(\d+))?(?:-[35]p)?$", re.I)


def mir_key(name):
    """(family, locus index or None) for precursor, mature or family names; None if not a miRNA name."""
    m = MIR.match(str(name).strip())
    return None if m is None else ((m.group(1).lower() + m.group(2).lower()), m.group(3))


def mirna_index(columns):
    """dataset miRNA names -> lookup(name) -> list of column positions."""
    fam, exact = defaultdict(list), {}
    for j, c in enumerate(columns):
        exact[c.lower()] = j
        k = mir_key(c)
        if k:
            fam[k[0]].append((j, k[1]))

    def lookup(name):
        n = str(name).strip().lower()
        if n in exact:
            return [exact[n]]
        k = mir_key(n)
        if k is None:
            return []
        cands = fam.get(k[0], [])
        if k[1] is not None:                      # a specific locus was named
            hit = [j for j, idx in cands if idx == k[1]]
            return hit or [j for j, idx in cands if idx is None]
        return [j for j, _ in cands]
    return lookup


# ------------------------------------------------------------------ sources -> long tables (name, partner, source)
def lnc_disease_records():
    dmap = pd.read_csv(SRC["disease_doid"], dtype=str)
    name2doid = dict(zip(dmap.disease_name.str.strip().str.lower(), dmap.doid.str.strip()))
    l2c = pd.read_csv(SRC["lnc2cancer"], encoding="utf-8-sig", dtype=str)
    a = pd.DataFrame({"lnc": l2c["name"], "disease": l2c["cancer type"], "source": "Lnc2Cancer3.0"})
    lrd = pd.read_csv(SRC["lncrnadisease"], encoding="utf-8-sig", dtype=str)
    lrd.columns = [c.strip() for c in lrd.columns]
    method = lrd["Validated Method//Prediction Method"].fillna("").str.strip()
    keep = (lrd["Species"].str.strip().str.lower() == "homo sapiens") \
        & (lrd["ncRNA Category"].str.strip().str.lower() == "lncrna") \
        & method.ne("") & ~method.str.upper().isin(PREDICTION_TOOLS)
    b = pd.DataFrame({"lnc": lrd.loc[keep, "ncRNA Symbol"], "disease": lrd.loc[keep, "Disease Name"],
                      "source": "LncRNADisease"})
    rec = pd.concat([a, b], ignore_index=True)
    rec["doid"] = rec.disease.str.strip().str.lower().map(name2doid)
    stats = {"Lnc2Cancer3.0 records": len(a), "LncRNADisease experimental records": len(b),
             "LncRNADisease records dropped (prediction / non-human / empty method)": int((~keep).sum()),
             "records with a DOID": int(rec.doid.notna().sum())}
    return rec.dropna(subset=["doid"]), stats


def lnc_mirna_records():
    enc = pd.read_csv(SRC["encori"], sep="\t", usecols=["miRNAname", "geneName", "geneType", "clipExpNum"])
    enc = enc[(enc.clipExpNum >= MIN_CLIP) & (enc.geneType == "lncRNA")]
    rec = enc.groupby(["geneName", "miRNAname"]).clipExpNum.max().reset_index()
    rec = rec.rename(columns={"geneName": "lnc", "miRNAname": "mirna"})
    rec["source"] = f"ENCORI(clipExpNum>={MIN_CLIP})"
    return rec, {"ENCORI CLIP-supported lncRNA-miRNA pairs": len(rec)}


def mirna_disease_records():
    hm = pd.read_csv(SRC["hmdd"], dtype=str)
    rec = hm[hm.doid.str.startswith("DOID", na=False)][["miRNA", "doid"]].rename(columns={"miRNA": "mirna"})
    rec["source"] = "HMDD4.0"
    return rec, {"HMDD v4 records": len(hm), "HMDD v4 records with a DOID": len(rec)}


# ------------------------------------------------------------------ build one dataset
def update(old, rows_of, cols_of, rec, row_key, col_key):
    """old: DataFrame (0/1). rows_of(name)->row positions, cols_of(name)->col positions. Returns new matrix + edges."""
    new = np.zeros(old.shape, dtype=np.int8)
    src = defaultdict(set)
    for r in rec.itertuples(index=False):
        for i in rows_of(getattr(r, row_key)):
            for j in cols_of(getattr(r, col_key)):
                new[i, j] = 1
                src[(i, j)].add(r.source)
    O = old.to_numpy().astype(np.int8)
    U = np.maximum(O, new)
    ii, jj = np.nonzero(U)
    edges = pd.DataFrame({"row": old.index[ii], "col": old.columns[jj], "in_old": O[ii, jj], "in_new": new[ii, jj],
                          "sources": [";".join(sorted(src.get((i, j), ()))) for i, j in zip(ii, jj)]})
    return pd.DataFrame(U.astype(int), index=old.index, columns=old.columns), edges


def load_do_ancestors():
    import networkx as nx
    import obonet
    G = obonet.read_obo(SRC["doid_obo"])            # edges child -> parent, so descendants() = ancestors
    return lambda d: nx.descendants(G, d) if d in G else set()


def build(dataset, canon, ld, lm, md, ancestors):
    base = os.path.join(REPO, "our_dataset", dataset)
    out = os.path.join(REPO, "our_dataset", f"{dataset}_{TAG}")
    read = lambda f: pd.read_csv(os.path.join(base, "interaction", f"{f}.csv"), index_col="0")
    lnc_di, lnc_mi, mi_di = read("lnc_di"), read("lnc_mi"), read("mi_di")

    lnc_pos = defaultdict(list)
    for i, n in enumerate(lnc_di.index):
        lnc_pos[canon(n)].append(i)
    lnc_rows = lambda n: lnc_pos.get(canon(n), [])
    dis_pos = {d: j for j, d in enumerate(lnc_di.columns)}
    propagate = PROPAGATE_TO_ANCESTORS[dataset]
    cache = {}

    def dis_cols(d):
        """Columns a record about disease d contributes to. Only diseases already in the dataset are used: a record
        about any other disease is ignored (it is NOT propagated to an in-dataset ancestor), so every new edge comes
        from a record that names an existing lncRNA/miRNA and an existing disease."""
        if d not in cache:
            if d not in dis_pos:
                cache[d] = []
            else:
                targets = {d} | (ancestors(d) if propagate else set())
                cache[d] = sorted(dis_pos[t] for t in targets if t in dis_pos)
        return cache[d]
    mi_cols = mirna_index(list(lnc_mi.columns))
    assert list(lnc_mi.columns) == list(mi_di.index) and list(lnc_di.columns) == list(mi_di.columns)
    ld = ld[ld.source.isin(LNC_DISEASE_SOURCES[dataset])]

    new_ld, e_ld = update(lnc_di, lnc_rows, dis_cols, ld, "lnc", "doid")
    new_lm, e_lm = update(lnc_mi, lnc_rows, mi_cols, lm, "lnc", "mirna")
    new_md, e_md = update(mi_di, mi_cols, dis_cols, md, "mirna", "doid")

    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(os.path.join(base, "basic"), os.path.join(out, "basic"))
    os.makedirs(os.path.join(out, "interaction"))
    os.makedirs(os.path.join(out, "provenance"))
    for name, m, e in (("lnc_di", new_ld, e_ld), ("lnc_mi", new_lm, e_lm), ("mi_di", new_md, e_md)):
        m.index.name = "0"
        m.to_csv(os.path.join(out, "interaction", f"{name}.csv"))
        e.to_csv(os.path.join(out, "provenance", f"{name}_edges.csv"), index=False)

    lines = [f"{dataset} -> {dataset}_{TAG}  (entities unchanged: {lnc_di.shape[0]} lncRNAs, {lnc_di.shape[1]} diseases, "
             f"{lnc_mi.shape[1]} miRNAs; new disease links propagated to DO ancestors: {propagate})",
             f"  new-release sources: lncRNA-disease {sorted(LNC_DISEASE_SOURCES[dataset])}, lncRNA-miRNA "
             f"ENCORI (clipExpNum>={MIN_CLIP}), miRNA-disease HMDD v4.0"]
    for name, e in (("lncRNA-disease", e_ld), ("lncRNA-miRNA", e_lm), ("miRNA-disease", e_md)):
        old_n, new_n = int(e.in_old.sum()), int(e.in_new.sum())
        both = int(((e.in_old == 1) & (e.in_new == 1)).sum())
        lines.append(f"  {name:15s} old {old_n:6d} | found in new releases {new_n:6d} (of which {both} were already known, "
                     f"old recovered {both / max(old_n, 1):.0%}) | added {new_n - both:6d} | total {len(e):6d}")
    ld_names = {canon(x) for x in ld.lnc}
    lines.append(f"  lncRNAs with >=1 record in the new lncRNA-disease releases: "
                 f"{sum(canon(n) in ld_names for n in lnc_di.index)}/{lnc_di.shape[0]}")
    lm_names = {canon(x) for x in lm.lnc}
    lines.append(f"  lncRNAs present in ENCORI CLIP-supported pairs: {sum(canon(n) in lm_names for n in lnc_di.index)}"
                 f"/{lnc_di.shape[0]}")
    with open(os.path.join(out, "build_report.txt"), "w") as f:
        f.write(__doc__ + "\n" + "\n".join(lines) + "\n")
    return lines


def main():
    canon = load_hgnc()
    ancestors = load_do_ancestors()
    ld, s1 = lnc_disease_records()
    lm, s2 = lnc_mirna_records()
    md, s3 = mirna_disease_records()
    for k, v in {**s1, **s2, **s3}.items():
        print(f"{k}: {v}")
    for ds in ("dataset1", "dataset2"):
        print("\n".join(build(ds, canon, ld, lm, md, ancestors)))


if __name__ == "__main__":
    main()
