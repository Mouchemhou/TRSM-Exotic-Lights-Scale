import os
import math
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import vector
import matplotlib.pyplot as plt

vector.register_awkward()

# ============================================================
# USER INPUT
# ============================================================

TREE_NAME = "Delphes"
OUTPUT_DIR = "cut_based_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LUMI_FB = 300.0
LUMI_PB = LUMI_FB * 1000.0

# Reconstructed h1 mass
MH = 20.13   #BP1
#MH = 24.84  #BP2
#MH = 21.60  #BP3
#MH = 30.16  #BP4
#MH = 22.27  #BP5
#MH = 36.70  #BP6
#MH = 37.83  #BP7
#MH = 20.92  #BP8
#MH = 20.38  #BP9

SAMPLES = {
    "signal": {
        "files": [
            "/path to/Sig_BP1/Events/run_01_decayed_1/tag_1_delphes_events.root",
        ],
        "label": 1,      
        "xsec_pb": 0.1,         #BP1 (17)
#        "xsec_pb": 0.06299,     # BP2   (28)
#        "xsec_pb": 0.05661,     # BP3  (32)
#        "xsec_pb": 0.03672,     # BP4  (23)
#        "xsec_pb": 0.02648,     # BP5  (56)
#        "xsec_pb": 0.02618,     # BP6  (29)
#         "xsec_pb": 0.01918,     # BP7  (28)
#        "xsec_pb": 0.01477,     # BP8  (140)
#        "xsec_pb":  0.008427,    # BP9  (147)
    },
    "ttjets": {
        "files": [
            "//ttbar_jets0/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//ttbar_jets1/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//ttbar_jets2/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//ttbar_jets3/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//ttbar_jets4/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//ttbar_jets5/Events/run_01_decayed_1/tag_1_delphes_events.root",
        ],
        "label": 0,
        "xsec_pb": 89.78,
    },


    "wjets": {
        "files": [
            "//Wjets0/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets1/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets2/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets3/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets4/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets5/Events/run_01_decayed_1/tag_1_delphes_events.root",
#            "//Wjets6/Events/run_01_decayed_1/tag_1_delphes_events.root",
        ],
        "label": 0,
        "xsec_pb": 475.3,         
    },
}

# ============================================================
# CUTS
# ============================================================
# For Nb >= 2 analyses, DeltaM_bb_min is usually the useful test.
# For Nb >= 4 analyses, Chi2_min is also meaningful.
CUTS = {
    "Nb_min": 2,
    "Nl_eq": 1,
    "Nj_eq": 2,
    "Nj_eq_min": 2,
    "Nj_eq_max": 2,
    "DeltaR_bb_min": 0.4,
    "DeltaR_bb_max": 1.6,

    # Use this for the 2b test
    "DeltaM_bb_min_max": 28,   # example: 15.0

    # Use this for the 4b test
    "chi2_max": None,            # example: 30.0
}

DELTAM_SCAN_VALUES = list(range(0,160)) #(DR) BP1(17), BP2(28)
#DELTAM_SCAN_VALUES = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34,35,36,37,38,39,40,50, 60, 70, 80] #(DR) BP3(32), BP4(23)
#DELTAM_SCAN_VALUES = [50, 52,53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67,68,69, 70, 75, 80, 85, 90, 95, 100] #(DR) BP5 (56)
#DELTAM_SCAN_VALUES = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40,45,50, 60, 70,80, 89, 100, 120, 140] # BP6 (29), BP7 (28)
#DELTAM_SCAN_VALUES = list(range(100, 160)) # BP8 (140-146), BP(140-147)
CHI2_SCAN_VALUES = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300]

# ============================================================
# HELPERS
# ============================================================

def safe_first(arr, default=-999.0):
    return ak.to_numpy(ak.fill_none(ak.firsts(arr), default))

def safe_nth(arr, n, default=-999.0):
    padded = ak.pad_none(arr, n + 1, axis=1, clip=True)
    return ak.to_numpy(ak.fill_none(padded[:, n], default))

def safe_count(arr):
    return ak.to_numpy(ak.num(arr, axis=1))

def build_p4(pt, eta, phi, mass):
    return ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": mass},
        with_name="Momentum4D",
    )

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.abs(dphi)

def delta_r_min(objects):
    pairs = ak.combinations(objects, 2, fields=["a", "b"])
    dr = pairs.a.deltaR(pairs.b)
    out = ak.min(dr, axis=1)
    out = ak.fill_none(out, -1.0)
    return ak.to_numpy(out)

def pair_mass(objects, i, j):
    padded = ak.pad_none(objects, max(i, j) + 1, axis=1, clip=True)
    oi = padded[:, i]
    oj = padded[:, j]
    mass = (oi + oj).mass
    mass = ak.fill_none(mass, -1.0)
    return ak.to_numpy(mass)

def invariant_mass_two_leading(objects):
    padded = ak.pad_none(objects, 2, axis=1, clip=True)
    obj0 = padded[:, 0]
    obj1 = padded[:, 1]
    mass = (obj0 + obj1).mass
    mass = ak.fill_none(mass, -1.0)
    return ak.to_numpy(mass)

def invariant_mass_three_leading(objects):
    padded = ak.pad_none(objects, 3, axis=1, clip=True)
    obj0 = padded[:, 0]
    obj1 = padded[:, 1]
    obj2 = padded[:, 2]
    mass = (obj0 + obj1 + obj2).mass
    mass = ak.fill_none(mass, -1.0)
    return ak.to_numpy(mass)

def invariant_mass_four_leading(objects):
    padded = ak.pad_none(objects, 4, axis=1, clip=True)
    obj0 = padded[:, 0]
    obj1 = padded[:, 1]
    obj2 = padded[:, 2]
    obj3 = padded[:, 3]
    mass = (obj0 + obj1 + obj2 + obj3).mass
    mass = ak.fill_none(mass, -1.0)
    return ak.to_numpy(mass)

def invariant_mass_bbj(bjets, nonbjets):
    bpad = ak.pad_none(bjets, 2, axis=1, clip=True)
    jpad = ak.pad_none(nonbjets, 1, axis=1, clip=True)
    b0 = bpad[:, 0]
    b1 = bpad[:, 1]
    j0 = jpad[:, 0]
    mass = (b0 + b1 + j0).mass
    mass = ak.fill_none(mass, -1.0)
    return ak.to_numpy(mass)

def pairing_label_from_index(idx_array):
    mapping = {
        0: "b1b2_b3b4",
        1: "b1b3_b2b4",
        2: "b1b4_b2b3",
        -1: "invalid",
    }
    return np.array([mapping.get(int(x), "invalid") for x in idx_array], dtype=object)

def compute_chi2_min_and_pairing(bjets, mh=MH):
    """
    4b test:
      chi2 = (mij - mh)^2 + (mkl - mh)^2
    using the 3 possible pairings of the 4 leading b-jets.
    Only valid for events with >= 4 b-jets.
    """
    padded = ak.pad_none(bjets, 4, axis=1, clip=True)

    b0 = padded[:, 0]
    b1 = padded[:, 1]
    b2 = padded[:, 2]
    b3 = padded[:, 3]

    m12 = (b0 + b1).mass
    m34 = (b2 + b3).mass
    chi2_0 = (m12 - mh) ** 2 + (m34 - mh) ** 2

    m13 = (b0 + b2).mass
    m24 = (b1 + b3).mass
    chi2_1 = (m13 - mh) ** 2 + (m24 - mh) ** 2

    m14 = (b0 + b3).mass
    m23 = (b1 + b2).mass
    chi2_2 = (m14 - mh) ** 2 + (m23 - mh) ** 2

    enough = ak.num(bjets, axis=1) >= 4

    chi2_all = ak.concatenate(
        [chi2_0[:, np.newaxis], chi2_1[:, np.newaxis], chi2_2[:, np.newaxis]],
        axis=1,
    )

    best_idx = ak.argmin(chi2_all, axis=1)
    chi2_min = ak.min(chi2_all, axis=1)

    chi2_min = ak.where(enough, chi2_min, -1.0)
    best_idx = ak.where(enough, best_idx, -1)

    return (
        ak.to_numpy(ak.fill_none(chi2_min, -1.0)),
        ak.to_numpy(ak.fill_none(best_idx, -1)),
    )

def compute_best_bb_pair(bjets, mh=MH):
    """
    For any event with >=2 b-jets, find the bb pair whose mass is closest to mh.

    Returns:
      best_delta = min |mij - mh|
      best_mass  = mass of the best pair
      best_label = label like b1b2, b1b3, ...
    """
    n_b = ak.num(bjets, axis=1)

    pairs = ak.combinations(bjets, 2, fields=["a", "b"])
    masses = (pairs.a + pairs.b).mass
    deltas = abs(masses - mh)

    best_delta = ak.min(deltas, axis=1)
    best_delta = ak.fill_none(best_delta, -1.0)

    best_idx = ak.argmin(deltas, axis=1)

    best_mass = ak.firsts(
        masses[ak.local_index(masses, axis=1) == best_idx[:, np.newaxis]],
        axis=1
    )
    best_mass = ak.fill_none(best_mass, -1.0)

    index_pairs = ak.argcombinations(bjets, 2, fields=["i", "j"])
    best_i = ak.firsts(
        index_pairs.i[ak.local_index(index_pairs.i, axis=1) == best_idx[:, np.newaxis]],
        axis=1
    )
    best_j = ak.firsts(
        index_pairs.j[ak.local_index(index_pairs.j, axis=1) == best_idx[:, np.newaxis]],
        axis=1
    )

    best_i = ak.fill_none(best_i, -1)
    best_j = ak.fill_none(best_j, -1)

    best_i_np = ak.to_numpy(best_i)
    best_j_np = ak.to_numpy(best_j)
    n_b_np = ak.to_numpy(n_b)

    labels = []
    for i, j, nb in zip(best_i_np, best_j_np, n_b_np):
        if nb < 2 or i < 0 or j < 0:
            labels.append("invalid")
        else:
            labels.append(f"b{i+1}b{j+1}")

    return (
        ak.to_numpy(best_delta),
        ak.to_numpy(best_mass),
        np.array(labels, dtype=object),
    )

def asimov_significance(S, B):
    if S <= 0 or B <= 0:
        return 0.0
    val = 2.0 * ((S + B) * math.log(1.0 + S / B) - S)
    return math.sqrt(max(val, 0.0))

# ============================================================
# READING + FEATURE BUILDING
# ============================================================

def load_features_from_root(root_file, label, sample_name, xsec_pb, event_weight):
    branches = [
        "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag",
        "Electron.PT", "Electron.Eta", "Electron.Phi",
        "Muon.PT", "Muon.Eta", "Muon.Phi",
        "MissingET.MET", "MissingET.Phi",
    ]

    tree = uproot.open(f"{root_file}:{TREE_NAME}")
    arrays = tree.arrays(branches, library="ak")

    jet_pt   = arrays["Jet.PT"]
    jet_eta  = arrays["Jet.Eta"]
    jet_phi  = arrays["Jet.Phi"]
    jet_mass = arrays["Jet.Mass"]
    jet_btag = arrays["Jet.BTag"]

    jets = build_p4(jet_pt, jet_eta, jet_phi, jet_mass)
    is_b = jet_btag > 0
    bjets = jets[is_b]
    nonbjets = jets[~is_b]

    ele_pt  = arrays["Electron.PT"]
    ele_eta = arrays["Electron.Eta"]
    ele_phi = arrays["Electron.Phi"]
    electrons = build_p4(ele_pt, ele_eta, ele_phi, ak.zeros_like(ele_pt))

    mu_pt  = arrays["Muon.PT"]
    mu_eta = arrays["Muon.Eta"]
    mu_phi = arrays["Muon.Phi"]
    muons = build_p4(mu_pt, mu_eta, mu_phi, ak.ones_like(mu_pt) * 0.105658)

    leptons = ak.concatenate([electrons, muons], axis=1)

    jets     = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    bjets    = bjets[ak.argsort(bjets.pt, axis=1, ascending=False)]
    nonbjets = nonbjets[ak.argsort(nonbjets.pt, axis=1, ascending=False)]
    leptons  = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

    met = safe_first(arrays["MissingET.MET"], default=0.0)
    met_phi = safe_first(arrays["MissingET.Phi"], default=0.0)

    # 4b test
    chi2_min, chi2_pairing = compute_chi2_min_and_pairing(bjets, mh=MH)
    chi2_pairing_label = pairing_label_from_index(chi2_pairing)

    # 2b test
    deltaM_bb_min, M_bb_best, best_bb_pair_label = compute_best_bb_pair(bjets, mh=MH)

    df = pd.DataFrame({
        "HT": ak.to_numpy(ak.sum(jets.pt, axis=1)),
        "ET": met,
        "Nb": safe_count(bjets),
        "Nl": safe_count(leptons),
        "Nj": safe_count(jets),

        "DeltaR_bb": delta_r_min(bjets),
        "DeltaR_jj": delta_r_min(jets),
        "DeltaR_ll": delta_r_min(leptons),

        "PT_b1": safe_nth(bjets.pt, 0),
        "PT_b2": safe_nth(bjets.pt, 1),
        "PT_b3": safe_nth(bjets.pt, 2),
        "PT_b4": safe_nth(bjets.pt, 3),

        "PT_j1": safe_nth(jets.pt, 0),
        "PT_j2": safe_nth(jets.pt, 1),
        "PT_l1": safe_nth(leptons.pt, 0),

        "M_b1b2": pair_mass(bjets, 0, 1),
        "M_b1b3": pair_mass(bjets, 0, 2),
        "M_b1b4": pair_mass(bjets, 0, 3),
        "M_b2b3": pair_mass(bjets, 1, 2),
        "M_b2b4": pair_mass(bjets, 1, 3),
        "M_b3b4": pair_mass(bjets, 2, 3),

        "M_bb": invariant_mass_two_leading(bjets),
        "M_bbb": invariant_mass_three_leading(bjets),
        "M_bbbb": invariant_mass_four_leading(bjets),
        "M_bbj": invariant_mass_bbj(bjets, nonbjets),

        # 4b test
        "Chi2_min": chi2_min,
        "Chi2_pairing": chi2_pairing,
        "Chi2_pairing_label": chi2_pairing_label,

        # 2b test
        "DeltaM_bb_min": deltaM_bb_min,
        "M_bb_best": M_bb_best,
        "Best_bb_pair_label": best_bb_pair_label,

        "DeltaPhi_MET_l": delta_phi(met_phi, safe_nth(leptons.phi, 0, default=0.0)),

        "label": label,
        "sample_name": sample_name,
        "source_file": os.path.basename(root_file),
        "xsec_pb": xsec_pb,
        "weight": event_weight,
        "n_gen_file": tree.num_entries,
    })

    return df

def build_dataset(samples_dict):
    dfs = []

    for sample_name, info in samples_dict.items():
        total_n_gen = 0
        for f in info["files"]:
            tree = uproot.open(f"{f}:{TREE_NAME}")
            total_n_gen += tree.num_entries

        event_weight = (info["xsec_pb"] * LUMI_PB) / float(total_n_gen)

        print(f"\nSample: {sample_name}")
        print(f"  total generated events = {total_n_gen}")
        print(f"  xsec_pb = {info['xsec_pb']}")
        print(f"  event weight = {event_weight}")

        for f in info["files"]:
            print(f"Loading {sample_name}: {f}")
            dfs.append(
                load_features_from_root(
                    root_file=f,
                    label=info["label"],
                    sample_name=sample_name,
                    xsec_pb=info["xsec_pb"],
                    event_weight=event_weight,
                )
            )

    df = pd.concat(dfs, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(-999.0)
    return df

# ============================================================
# PLOTTING
# ============================================================

PLOT_CONFIGS = {
    "HT": (40, 0, 1500),
    "ET": (40, 0, 500),
    "Nb": (8, -0.5, 7.5),
    "Nl": (6, -0.5, 5.5),
    "Nj": (12, -0.5, 11.5),
    "DeltaR_bb": (30, 0, 5),
    "DeltaR_jj": (30, 0, 5),
    "DeltaR_ll": (30, 0, 5),
    "PT_b1": (40, 0, 500),
    "PT_b2": (40, 0, 500),
    "PT_b3": (40, 0, 500),
    "PT_b4": (40, 0, 500),
    "PT_j1": (40, 0, 500),
    "PT_j2": (40, 0, 500),
    "PT_l1": (40, 0, 300),

    "M_b1b2": (20, 0, 160),
    "M_b1b3": (20, 0, 160),
    "M_b1b4": (20, 0, 160),
    "M_b2b3": (20, 0, 160),
    "M_b2b4": (20, 0, 160),
    "M_b3b4": (20, 0, 160),

    "M_bb": (20, 0, 160),
    "M_bbj": (20, 0, 300),
    "M_bbb": (20, 0, 300),
    "M_bbbb": (20, 0, 300),

    "Chi2_min": (40, 0, 300),
    "DeltaM_bb_min": (40, 0, 80),
    "M_bb_best": (20, 0, 160),

    "DeltaPhi_MET_l": (20, 0, math.pi),
}

def plot_variable(df, varname, bins, xmin, xmax, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(7, 5))

    for sample_name in df["sample_name"].unique():
        sub = df[df["sample_name"] == sample_name]

        vals = np.asarray(sub[varname], dtype=float)
        weights = np.asarray(sub["weight"], dtype=float)

        mask = np.isfinite(vals) & (vals > -900)
        vals = vals[mask]
        weights = weights[mask]

        if len(vals) == 0:
            continue

        plt.hist(
            vals,
            bins=bins,
            range=(xmin, xmax),
            weights=weights,
            density=True,
            histtype="step",
            linewidth=2,
            label=sample_name,
        )

    plt.xlabel(varname)
    plt.ylabel("Normalized entries")
    plt.title(varname)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{varname}.png"), dpi=180)
    plt.close()

def plot_all_variables(df, outdir):
    for varname, cfg in PLOT_CONFIGS.items():
        plot_variable(df, varname, cfg[0], cfg[1], cfg[2], outdir)

# ============================================================
# SCANS
# ============================================================

def scan_deltaM_bb_min(df, cuts, scan_values):
    rows = []
    baseline = (
        (df["Nb"] >= cuts["Nb_min"]) &
        (df["Nl"] == cuts["Nl_eq"]) &
        (df["Nj"] >= cuts["Nj_eq_min"]) &  
        (df["Nj"] <= cuts["Nj_eq_max"]) &
        (df["DeltaR_bb"] >= cuts["DeltaR_bb_min"]) &
        (df["DeltaR_bb"] <= cuts["DeltaR_bb_max"])
    )

    for cutval in scan_values:
        mask = baseline & (df["DeltaM_bb_min"] >= 0) & (df["DeltaM_bb_min"] < cutval)
        sub = df[mask]

        S = sub.loc[sub["label"] == 1, "weight"].sum()
        B = sub.loc[sub["label"] == 0, "weight"].sum()
        Z = asimov_significance(S, B)

        rows.append({
            "DeltaM_bb_min_cut": cutval,
            "S_expected": S,
            "B_expected": B,
            "S_xsec_eff_pb": S / LUMI_PB,
            "B_xsec_eff_pb": B / LUMI_PB,
            "Z_A": Z,
        })

    return pd.DataFrame(rows)

# ============================================================
# CUT-BASED ANALYSIS
# ============================================================

def apply_cut_based_selection(df, cuts):
    mask = (
        (df["Nb"] >= cuts["Nb_min"]) &
        (df["Nl"] == cuts["Nl_eq"]) &
        (df["Nj"] >= cuts["Nj_eq_min"]) &  
        (df["Nj"] <= cuts["Nj_eq_max"]) &
        (df["DeltaR_bb"] >= cuts["DeltaR_bb_min"]) &
        (df["DeltaR_bb"] <= cuts["DeltaR_bb_max"])
    )

    if cuts["chi2_max"] is not None:
        mask = mask & (df["Chi2_min"] >= 0) & (df["Chi2_min"] < cuts["chi2_max"])

    if cuts["DeltaM_bb_min_max"] is not None:
        mask = mask & (df["DeltaM_bb_min"] >= 0) & (df["DeltaM_bb_min"] < cuts["DeltaM_bb_min_max"])

    return df[mask].copy()

def compute_yields_and_significance(df_sel):
    sig_df = df_sel[df_sel["label"] == 1]
    bkg_df = df_sel[df_sel["label"] == 0]

    S = sig_df["weight"].sum()
    B = bkg_df["weight"].sum()
    Z = asimov_significance(S, B)

    return S, B, Z

def cutflow(df, cuts):
    rows = []

    selections = [
        ("All events", np.ones(len(df), dtype=bool)),

        (
            f"Nb >= {cuts['Nb_min']}",
            (df["Nb"] >= cuts["Nb_min"]).values
        ),

        (
            f"Nb >= {cuts['Nb_min']} and Nl == {cuts['Nl_eq']}",
            (
                (df["Nb"] >= cuts["Nb_min"]) &
                (df["Nl"] == cuts["Nl_eq"])
            ).values
        ),

        (
            f"Nb >= {cuts['Nb_min']} and Nl == {cuts['Nl_eq']} and {cuts['Nj_eq_min']} <= Nj <= {cuts['Nj_eq_max']}",
            (
                (df["Nb"] >= cuts["Nb_min"]) &
                (df["Nl"] == cuts["Nl_eq"]) &
                (df["Nj"] >= cuts["Nj_eq_min"]) &
                (df["Nj"] <= cuts["Nj_eq_max"])
            ).values
        ),

        (
            f"... and {cuts['DeltaR_bb_min']} <= DeltaR_bb <= {cuts['DeltaR_bb_max']}",
            (
                (df["Nb"] >= cuts["Nb_min"]) &
                (df["Nl"] == cuts["Nl_eq"]) &
                (df["Nj"] >= cuts["Nj_eq_min"]) &
                (df["Nj"] <= cuts["Nj_eq_max"]) &
                (df["DeltaR_bb"] >= cuts["DeltaR_bb_min"]) &
                (df["DeltaR_bb"] <= cuts["DeltaR_bb_max"])
            ).values
        ),
    ]

    if cuts["chi2_max"] is not None:
        selections.append((
            f"... and Chi2_min < {cuts['chi2_max']}",
            (
                (df["Nb"] >= cuts["Nb_min"]) &
                (df["Nl"] == cuts["Nl_eq"]) &
                (df["Nj"] >= cuts["Nj_eq_min"]) &
                (df["Nj"] <= cuts["Nj_eq_max"]) &
                (df["DeltaR_bb"] >= cuts["DeltaR_bb_min"]) &
                (df["DeltaR_bb"] <= cuts["DeltaR_bb_max"]) &
                (df["Chi2_min"] >= 0) &
                (df["Chi2_min"] < cuts["chi2_max"])
            ).values
        ))

    if cuts["DeltaM_bb_min_max"] is not None:
        selections.append((
            f"... and DeltaM_bb_min < {cuts['DeltaM_bb_min_max']}",
            (
                (df["Nb"] >= cuts["Nb_min"]) &
                (df["Nl"] == cuts["Nl_eq"]) &
                (df["Nj"] >= cuts["Nj_eq_min"]) &
                (df["Nj"] <= cuts["Nj_eq_max"]) &
                (df["DeltaR_bb"] >= cuts["DeltaR_bb_min"]) &
                (df["DeltaR_bb"] <= cuts["DeltaR_bb_max"]) &
                (df["DeltaM_bb_min"] >= 0) &
                (df["DeltaM_bb_min"] < cuts["DeltaM_bb_min_max"])
            ).values
        ))

    for name, mask in selections:
        sub = df[mask]
        S = sub.loc[sub["label"] == 1, "weight"].sum()
        B = sub.loc[sub["label"] == 0, "weight"].sum()
        Z = asimov_significance(S, B)

        rows.append({
            "cut": name,
            "S_expected": S,
            "B_expected": B,
            "S_xsec_eff_pb": S / LUMI_PB,
            "B_xsec_eff_pb": B / LUMI_PB,
            "Z_A": Z,
        })

    cf = pd.DataFrame(rows)

    # cumulative efficiencies relative to first row
    S0 = cf.loc[0, "S_expected"]
    B0 = cf.loc[0, "B_expected"]

    cf["sig_eff_cum"] = cf["S_expected"] / S0 if S0 > 0 else 0.0
    cf["bkg_eff_cum"] = cf["B_expected"] / B0 if B0 > 0 else 0.0

    # step efficiencies relative to previous row
    cf["sig_eff_step"] = 1.0
    cf["bkg_eff_step"] = 1.0

    for i in range(1, len(cf)):
        prevS = cf.loc[i - 1, "S_expected"]
        prevB = cf.loc[i - 1, "B_expected"]

        cf.loc[i, "sig_eff_step"] = cf.loc[i, "S_expected"] / prevS if prevS > 0 else 0.0
        cf.loc[i, "bkg_eff_step"] = cf.loc[i, "B_expected"] / prevB if prevB > 0 else 0.0

    return cf
# ============================================================
# MAIN
# ============================================================

def main():
    print("Building dataset...")
    df = build_dataset(SAMPLES)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_dataset.csv"), index=False)

    print("\nPlotting variables before cuts...")
    plot_all_variables(df, os.path.join(OUTPUT_DIR, "plots_before_cuts"))

    print("\nBest 4b pairing summary:")
    print(df["Chi2_pairing_label"].value_counts())

    print("\nBest bb pair summary (DeltaMmin test):")
    print(df["Best_bb_pair_label"].value_counts())

    print("\nScanning DeltaM_bb_min cut...")
    deltaM_scan_df = scan_deltaM_bb_min(df, CUTS, DELTAM_SCAN_VALUES)
    deltaM_scan_df.to_csv(os.path.join(OUTPUT_DIR, "deltaM_scan.csv"), index=False)
    print(deltaM_scan_df.to_string(index=False))

    print("\nApplying cut-based analysis...")
    df_cut = apply_cut_based_selection(df, CUTS)
    df_cut.to_csv(os.path.join(OUTPUT_DIR, "dataset_after_cuts.csv"), index=False)

    S, B, Z = compute_yields_and_significance(df_cut)

    cf = cutflow(df, CUTS)
    cf.to_csv(os.path.join(OUTPUT_DIR, "cutflow.csv"), index=False)

    print("\n============================================================")
    print("CUT-BASED FINAL RESULTS")
    print("============================================================")
    print(f"Luminosity                  = {LUMI_FB} fb^-1")
    print(f"Using mH                    = {MH}")
    print(f"Selection                   = Nb >= {CUTS['Nb_min']}, Nl == {CUTS['Nl_eq']}, {CUTS['Nj_eq_min']} <= Nj <= {CUTS['Nj_eq_max']}")
    print(f"DeltaR_bb cut               = [{CUTS['DeltaR_bb_min']}, {CUTS['DeltaR_bb_max']}]")
    print(f"Chi2_min cut                = {CUTS['chi2_max']}")
    print(f"DeltaM_bb_min cut           = {CUTS['DeltaM_bb_min_max']}")
    print(f"Signal yield S              = {S:.6f}")
    print(f"Background yield B          = {B:.6f}")
    print(f"Signal eff. xsec [pb]       = {S / LUMI_PB:.8f}")
    print(f"Background eff. xsec [pb]   = {B / LUMI_PB:.8f}")
    print(f"Asimov significance         = {Z:.6f}")

    # total efficiencies = last cut relative to all events
    total_sig_eff = cf.iloc[-1]["sig_eff_cum"]
    total_bkg_eff = cf.iloc[-1]["bkg_eff_cum"]

    print(f"Total signal efficiency     = {total_sig_eff:.6f} ({100.0*total_sig_eff:.3f}%)")
    print(f"Total background efficiency = {total_bkg_eff:.6e} ({100.0*total_bkg_eff:.6f}%)")

    print("\nCutflow:")
    print(cf.to_string(index=False))

    print("\nExamples of DeltaMmin test for events with Nb >= 2:")
    cols = ["Nb", "M_bb_best", "DeltaM_bb_min", "Best_bb_pair_label"]
    print(df[df["Nb"] >= 2][cols].head(10).to_string(index=False))

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
