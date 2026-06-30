# VH TRSM Collider Analysis

This repository contains the scripts and analysis workflow used to study associated production of a scalar in the Two Real Singlet Model:

```text
pp → Vh₂ → Vh₁h₁ → Vbb̄bb̄
```

where **V = W± or Z**. The repository includes scripts for ScannerS, HiggsTools, MadGraph5_aMC@NLO, Pythia, Delphes, MadAnalysis5, and Python-based event analysis.

## Repository structure

### ScannerS_HiggsTools

Contains the scripts used to scan the TRSM parameter space and test theoretical and experimental constraints using ScannerS and HiggsTools.

### UFO_and_ParamCard-generator

Contains the UFO model and the TwoSinglet code used to generate MadGraph parameter cards.

The external TwoSinglet code can be added as a Git submodule:

```bash
git submodule add https://gitlab.com/apapaefs/twosinglet.git UFO_and_ParamCard-generator/twosinglet-master
```

### Benchmark_ParamCards

Contains the MadGraph parameter cards for the selected benchmark points.

```
BP1
BP2
BP3
...
```

### MadGraph_Delphes

Contains MadGraph process cards and Delphes cards used to generate and simulate signal and background events.

Example signal process:

```text
p p > h2 w+
h2 > h1 h1
h1 > b b~
w+ > l+ vl
```

### MadAnalysis

Contains MadAnalysis5 scripts used for cut-based analysis, event selection, cutflow tables, and validation plots.

### Python_Analysis

Contains Python codes used to analyze signal and background ROOT files after Delphes simulation.

This includes:

* ROOT file reading with uproot
* object selection
* feature construction
* cut-based analysis
* plotting
* significance calculation


## Suggested workflow

1. Run ScannerS to generate allowed TRSM points.
2. Apply HiggsTools constraints.
3. Select benchmark points.
4. Generate MadGraph parameter cards using the TwoSinglet code.
5. Generate signal and background events with MadGraph.
6. Shower and hadronize events with Pythia.
7. Simulate detector effects with Delphes.
8. Analyze Delphes ROOT files using MadAnalysis5 and Python.
9. Produce cutflows, plots, and significance results.

## Citation

If you use the TwoSinglet code, please cite the original authors and repository:

Andreas Papaefstathiou, Tania Robens, and Gilberto Tetlalmatzi-Xolocotzi, Two Real Singlet Model implementation.

GitLab repository:

```text
https://gitlab.com/apapaefs/twosinglet
```



