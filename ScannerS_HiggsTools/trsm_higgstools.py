import Higgs.predictions as HP
import Higgs.bounds as HB
import Higgs.signals as HS
import numpy as np
import pandas as pd
from tqdm import tqdm
from twors_higgstools_functions import *   # probably need much less as in principle can read in everything from the .tsv output from scanners

pred = HP.Predictions() # create the model predictions
bounds = HB.Bounds('//hbdataset-v1.7') # load HB dataset
signals = HS.Signals('//hsdataset-v1.1') # load HS dataset

## Set Higgs Prediction

h1 = pred.addParticle(HP.BsmParticle("h1", "neutral", "even"))
h2 = pred.addParticle(HP.BsmParticle("h2", "neutral", "even"))
h3 = pred.addParticle(HP.BsmParticle("h3", "neutral", "even"))

h1.setMassUnc(0)
h2.setMassUnc(0)
h3.setMassUnc(0)

### Define functions for reading ScannerS output into HiggsPredictions and for running HiggsBounds + HiggsSignals
class DecayError(Exception): pass

# run HB and HS for one parameter point pt with couplings cpl
def run_higgstools(cpl, pt):
    h1.setMass(pt['mH1'])
    h2.setMass(pt['mH2'])
    h3.setMass(pt['mH3'])
    
    set_h1_properties(cpl[0], pt)
    set_h2_properties(cpl[1], pt)
    set_h3_properties(cpl[2], pt)
    res = bounds(pred)
    chisq = signals(pred)
    return res, chisq

# set properties of the h1 boson
def set_h1_properties(dc, pt):
#    cpls = HP.NeutralEffectiveCouplings()
#    cpls.tt = dc['tt']
#    cpls.bb = dc['bb']
#    cpls.ZZ = dc['ZZ']
#    cpls.WW = dc['WW']

    if pt['mH1'] < 150:
        HP.effectiveCouplingInput(
            h1,
            HP.scaledSMlikeEffCouplings(pt['R11']),
            reference=HP.ReferenceModel.SMHiggsEW)
    else:
        HP.effectiveCouplingInput(
            h1,
            HP.scaledSMlikeEffCouplings(pt['R11']))

    h1.setCxn('LHC13', 'ggH', h1.cxn('LHC13', "ggH"))
    h1.setCxn('LHC13', 'bbH', h1.cxn('LHC13', "bbH"))
    w = pt['Wh1']
    h1.setDecayWidth('gg', pt['BRh1gg'] * w)
    h1.setDecayWidth('bb', pt['BRh1bb'] * w)
    h1.setDecayWidth('tautau', pt['BRh1tata'] * w)
    h1.setDecayWidth('cc', pt['BRh1cc'] * w)
    h1.setDecayWidth('ss', pt['BRh1ss'] * w)
    h1.setDecayWidth('mumu', pt['BRh1mumu'] * w)
    h1.setDecayWidth('gamgam', pt['BRh1gaga'] * w)
    h1.setDecayWidth('Zgam', pt['BRh1ZGA'] * w)
    h1.setDecayWidth('WW', pt['BRh1WW'] * w)
    h1.setDecayWidth('ZZ', pt['BRh1ZZ'] * w)
    if abs(h1.totalWidth() - w) > 1e-3:
        raise DecayError("Missing decay channel for particle h1.")

# set properties of the H boson
def set_h2_properties(dc, pt):
#    cpls = HP.NeutralEffectiveCouplings()
#    cpls.tt = dc['tt']
#    cpls.bb = dc['bb']
#    cpls.ZZ = dc['ZZ']
#    cpls.WW = dc['WW']

    if pt['mH2'] < 150:
        HP.effectiveCouplingInput(
            h2,
            HP.scaledSMlikeEffCouplings(pt['R21']),
            reference=HP.ReferenceModel.SMHiggsEW)
    else:
        HP.effectiveCouplingInput(
            h2,
            HP.scaledSMlikeEffCouplings(pt['R21']))

#    h2.setCxn('LHC13', 'ggH', pt['x_H2_gg'])
#    h2.setCxn('LHC13', 'bbH', pt['x_H2_bb'])
    h2.setCxn('LHC13', 'ggH', h2.cxn('LHC13', "ggH"))
#    h2.setCxn('LHC13', 'ggH', 2.46)
    h2.setCxn('LHC13', 'bbH', h2.cxn('LHC13', "bbH"))
    w = pt['Wh2']
    h2.setDecayWidth('gg', pt['BRh2gg'] * w)
    h2.setDecayWidth('WW', pt['BRh2WW'] * w)
    h2.setDecayWidth('ZZ', pt['BRh2ZZ'] * w)
    h2.setDecayWidth('gamgam', pt['BRh2gaga'] * w)
    h2.setDecayWidth('tt', pt['BRh2tt'] * w)
    h2.setDecayWidth('bb', pt['BRh2bb'] * w)
    h2.setDecayWidth('cc', pt['BRh2cc'] * w)
    h2.setDecayWidth('ss', pt['BRh2ss'] * w)
    h2.setDecayWidth('tautau', pt['BRh2mumu'] * w)
    h2.setDecayWidth('h1', 'h1', pt['BRh2h1h1'] * w)
    if abs(h2.totalWidth() - w) > 1e-3:
        raise DecayError("Missing decay channel for particle h2.")

# set properties of the h3 boson
def set_h3_properties(dc, pt):
#    cpls = HP.NeutralEffectiveCouplings()
#    cpls.tt = dc['tt']
#    cpls.bb = dc['bb']
#    cpls.ZZ = dc['ZZ']
#    cpls.WW = dc['WW']

    if pt['mH3'] < 150:
        HP.effectiveCouplingInput(
            h3,
            HP.scaledSMlikeEffCouplings(pt['R31']),
            reference=HP.ReferenceModel.SMHiggsEW)
    else:
        HP.effectiveCouplingInput(
            h3,
            HP.scaledSMlikeEffCouplings(pt['R31']))

#    h3.setCxn('LHC13', 'ggH', pt['x_H3_gg'])
#    h3.setCxn('LHC13', 'bbH', pt['x_H3_bb'])
#    h3.setCxn('LHC13', 'ggH', h3.cxn('LHC13', "ggH"))
    h3.setCxn('LHC13', 'ggH', 45.2)

    h3.setCxn('LHC13', 'bbH', h3.cxn('LHC13', "bbH"))
    w = pt['Wh3']
    h3.setDecayWidth('gg', pt['BRh3gg'] * w)
    h3.setDecayWidth('gamgam', pt['BRh3gaga'] * w)
    h3.setDecayWidth('tt', pt['BRh3tt'] * w)
    h3.setDecayWidth('bb', pt['BRh3bb'] * w)
    h3.setDecayWidth('cc', pt['BRh3cc'] * w)
    h3.setDecayWidth('ss', pt['BRh3ss'] * w)
    h3.setDecayWidth('tautau', pt['BRh3tata'] * w)
    h3.setDecayWidth('h1', 'h1', pt['BRh3h1h1'] * w)
    h3.setDecayWidth('h1', 'h2', pt['BRh3h1h2'] * w)
    h3.setDecayWidth('h2', 'h2', pt['BRh3h2h2'] * w)
    if abs(h3.totalWidth() - w) > 1e-3:
        raise DecayError("Missing decay channel for particle h3.")

# read ScannerS output file
def read_scanners_output(path):
    df = pd.read_csv(path, delimiter='\t',index_col=[0])
    dcs = []
    for i, r in df.iterrows():
        dc = {}
        dc['mH1'] = r['mH1']
        dc['mH2'] = r['mH2']
        dc['mH3'] = r['mH3']
        dc['thetahS'] = r['thetahS']
        dc['thetahX'] = r['thetahX']
        dc['thetaSX'] = r['thetaSX']
        dc['R11'] = r['R11']
        dc['R21'] = r['R21']
        dc['R31'] = r['R31']

        # h1 Branching Ratio
        dc['BRh1WW'] = r['b_H1_WW']
        dc['BRh1ZZ'] = r['b_H1_ZZ']
        dc['BRh1ZGA'] = r['b_H1_Zgam']
        dc['BRh1bb'] = r['b_H1_bb']
        dc['BRh1cc'] = r['b_H1_cc']
        dc['BRh1gaga'] = r['b_H1_gamgam']
        dc['BRh1gg'] = r['b_H1_gg']
        dc['BRh1mumu'] = r['b_H1_mumu']
        dc['BRh1ss'] = r['b_H1_ss']
        dc['BRh1tata'] = r['b_H1_tautau']
        dc['BRh1tt'] = r['b_H1_tt']
        
        # h2 Branging Ratio
        dc['BRh2h1h1'] = r['b_H2_H1H1']
        dc['BRh2WW'] = r['b_H2_WW']
        dc['BRh2ZZ'] = r['b_H2_ZZ']
        dc['BRh2ZGA'] = r['b_H2_Zgam']
        dc['BRh2bb'] = r['b_H2_bb']
        dc['BRh2cc'] = r['b_H2_cc']
        dc['BRh2gaga'] = r['b_H2_gamgam']
        dc['BRh2gg'] = r['b_H2_gg']
        dc['BRh2mumu'] = r['b_H2_mumu']
        dc['BRh2ss'] = r['b_H2_ss']
        dc['BRh2tata'] = r['b_H2_tautau']
        dc['BRh2tt'] = r['b_H2_tt']
        ## h3 Branching Ratio
        dc['BRh3h1h1'] = r['b_H3_H1H1']
        dc['BRh3h1h2'] = r['b_H3_H1H2']
        dc['BRh3h2h2'] = r['b_H3_H2H2']
        dc['BRh3WW'] = r['b_H3_WW']
        dc['BRh3ZZ'] = r['b_H3_ZZ']
        dc['BRh3ZGA'] = r['b_H3_Zgam']
        dc['BRh3bb'] = r['b_H3_bb']
        dc['BRh3cc'] = r['b_H3_cc']
        dc['BRh3gaga'] = r['b_H3_gamgam']
        dc['BRh3gg'] = r['b_H3_gg']
        dc['BRh3mumu'] = r['b_H3_mumu']
        dc['BRh3ss'] = r['b_H3_ss']
        dc['BRh3tata'] = r['b_H3_tautau']
        dc['BRh3tt'] = r['b_H3_tt']
        ## Coupling
        dc['c_H1H1H1'] = r['c_H1H1H1']
        dc['c_H1H1H2'] = r['c_H1H1H2']
        dc['c_H1H1H3'] = r['c_H1H1H3']
        dc['c_H1H2H2'] = r['c_H1H2H2']
        dc['c_H1H2H3'] = r['c_H1H2H3']
        dc['c_H1H3H3'] = r['c_H1H3H3']
        dc['c_H2H2H2'] = r['c_H2H2H2']
        dc['c_H2H2H3'] = r['c_H2H2H3']
        dc['c_H2H3H3'] = r['c_H2H3H3']
        dc['c_H3H3H3'] = r['c_H3H3H3']
        ## =========
        dc['x_H1W'] = r['x_H1W']
        dc['x_H1Z'] = r['x_H1Z']
        dc['x_H1_bb'] = r['x_H1_bb']
        dc['x_H1_gg'] = r['x_H1_gg']
        dc['x_H1_vbf'] = r['x_H1_vbf']
        dc['x_H1tt'] = r['x_H1tt']
        dc['x_H2W'] = r['x_H2W']
        dc['x_H2Z'] = r['x_H2Z']
        dc['x_H2_bb'] = r['x_H2_bb']
        dc['x_H2_gg'] = r['x_H2_gg']
        dc['x_H2_vbf'] = r['x_H2_vbf']
        dc['x_H2tt'] = r['x_H2tt']
        dc['x_H3W'] = r['x_H3W']
        dc['x_H3Z'] = r['x_H3Z']
        dc['x_H3_bb'] = r['x_H3_bb']
        dc['x_H3_gg'] = r['x_H3_gg']
        dc['x_H3_vbf'] = r['x_H3_vbf']
        dc['x_H3tt'] = r['x_H3tt']
        # Largeur Total
        dc['Wh1'] = r['w_H1']
        dc['Wh2'] = r['w_H2']
        dc['Wh3'] = r['w_H3']
        dcs.append(dc)
    return dcs

# calculate effective couplings in type 1
def calc_effective_couplings(aa, bb, cc):

    cplh1 = {
        'tt': aa,
        'bb': aa,
        'ZZ': aa,
        'WW': aa}
    cplh2 = {
        'tt': bb,
        'bb': bb,
        'ZZ': bb,
        'WW': bb}
    cplh3 = {
        'tt': cc,
        'bb': cc,
        'ZZ': cc,
        'WW': cc}
    return [cplh1, cplh2, cplh3]
    
# process dataset and save output to file
def process_data(dataset):
    data = []
    jj= 0
    for point in tqdm(dataset):
        mH1 = point['mH1']
        mH2 = point['mH2']
        mH3 = point['mH3']
        R11 = point['R11']
        R21 = point['R21']
        R31 = point['R31']
        # h1 Branching Ratio
        BRh1WW = point['BRh1WW']
        BRh1ZZ = point['BRh1ZZ']
        BRh1ZGA =point['BRh1ZGA']
        BRh1bb = point['BRh1bb']
        BRh1cc = point['BRh1cc']
        BRh1gaga = point['BRh1gaga']
        BRh1gg = point['BRh1gg']
        BRh1mumu = point['BRh1mumu']
        BRh1ss = point['BRh1ss']
        BRh1tata = point['BRh1tata']
        BRh1tt = point['BRh1tt']
        # h2 Branging Ratio
        BRh2h1h1 = point['BRh2h1h1']
        BRh2WW = point['BRh2WW']
        BRh2ZZ = point['BRh2ZZ']
        BRh2ZGA = point['BRh2ZGA']
        BRh2bb = point['BRh2bb']
        BRh2cc = point['BRh2cc']
        BRh2gaga = point['BRh2gaga']
        BRh2gg = point['BRh2gg']
        BRh2mumu = point['BRh2mumu']
        BRh2ss = point['BRh2ss']
        BRh2tata = point['BRh2tata']
        BRh2tt = point['BRh2tt']
        ## h3 Branching Ratio
        BRh3h1h1 = point['BRh3h1h1']
        BRh3h1h2 = point['BRh3h1h2']
        BRh3h2h2 = point['BRh3h2h2']
        BRh3WW = point['BRh3WW']
        BRh3ZZ = point['BRh3ZZ']
        BRh3ZGA = point['BRh3ZGA']
        BRh3bb = point['BRh3bb']
        BRh3cc = point['BRh3cc']
        BRh3gaga = point['BRh3gaga']
        BRh3gg = point['BRh3gg']
        BRh3mumu = point['BRh3mumu']
        BRh3ss = point['BRh3ss']
        BRh3tata = point['BRh3tata']
        BRh3tt = point['BRh3tt']
        ## Coupling
        c_H1H1H1 = point['c_H1H1H1']
        c_H1H1H2 = point['c_H1H1H2']
        c_H1H1H3 = point['c_H1H1H3']
        c_H1H2H2 = point['c_H1H2H2']
        c_H1H2H3 = point['c_H1H2H3']
        c_H1H3H3 = point['c_H1H3H3']
        c_H2H2H2 = point['c_H2H2H2']
        c_H2H2H3 = point['c_H2H2H3']
        c_H2H3H3 = point['c_H2H3H3']
        c_H3H3H3 = point['c_H3H3H3']
        ## ========= Cross-sections
        x_H1W = point['x_H1W']
        x_H1Z = point['x_H1Z']
        x_H1_bb = point['x_H1_bb']
        x_H1_gg = point['x_H1_gg']
        x_H1_vbf = point['x_H1_vbf']
        x_H1tt = point['x_H1tt']
        ##=========================
        x_H2W = point['x_H2W']
        x_H2Z = point['x_H2Z']
        x_H2_bb = point['x_H2_bb']
        x_H2_gg = point['x_H2_gg']
        x_H2_vbf = point['x_H2_vbf']
        x_H2tt = point['x_H2tt']
        ##=========================
        x_H3W = point['x_H3W']
        x_H3Z = point['x_H3Z']
        x_H3_bb = point['x_H3_bb']
        x_H3_gg = point['x_H3_gg']
        x_H3_vbf = point['x_H3_vbf']
        x_H3tt = point['x_H3tt']
         # Largeur Total
        Wh1 = point['Wh1']
        Wh2 = point['Wh2']
        Wh3 = point['Wh3']
        cpl = calc_effective_couplings(R11, R21, R31)
        reshb, Chisq = run_higgstools(cpl, point)
        if True:#reshb.selectedLimits['h1'].obsRatio()<=1.0 and reshb.selectedLimits['h2'].obsRatio() <= 1. and reshb.selectedLimits['h3'].obsRatio() <= 1.0:
            jj=jj+1 
            print(Chisq)
            print(reshb)       
            data.append({
                'mH1' : mH1,
                'mH2' : mH2,
                'mH3' : mH3,
                'chisq': Chisq,
                'BRh1WW': BRh1WW,
                'BRh1ZZ': BRh1ZZ,
                'BRh1ZGA':BRh1ZGA,
                'BRh1bb': BRh1bb,
                'BRh1cc': BRh1cc,
                'BRh1gaga': BRh1gaga,
                'BRh1gg' : BRh1gg,
                'BRh1mumu': BRh1mumu,
                'BRh1ss': BRh1ss,
                'BRh1tata':BRh1tata,
                'BRh1tt':BRh1tt,
                #Branching ratios of h2
                'BRh2h1h1':BRh2h1h1,
                'BRh2WW': BRh2WW,
                'BRh2ZZ': BRh2ZZ,
                'BRh2ZGA': BRh2ZGA,
                'BRh2bb' : BRh2bb,
                'BRh2cc' : BRh2cc,
                'BRh2gaga':BRh2gaga,
                'BRh2gg':BRh2gg,
                'BRh2mumu':BRh2mumu,
                'BRh2ss': BRh2ss,
                'BRh2tata': BRh2tata,
                'BRh1tt':BRh2tt,
                #Branching ratios h3.
                'BRh3h1h1':BRh3h1h1,
                'BRh3h1h2':BRh3h1h2,
                'BRh3h2h2':BRh3h2h2,
                'BRh3WW': BRh3WW,
                'BRh3ZZ': BRh3ZZ,
                'BRh3ZGA': BRh3ZGA,
                'BRh3bb': BRh3bb,
                'BRh3cc': BRh3cc,
                'BRh3gaga': BRh3gaga,
                'BRh3gg': BRh3gg,
                'BRh3mumu': BRh3mumu,
                'BRh3ss': BRh3ss,
                'BRh3tata': BRh3tata,
                'BRh3tt': BRh3tata,
                #Scalar-scalar-Coupling.
                'c_H1H1H1': c_H1H1H1,
                'c_H1H1H2': c_H1H1H2,
                'c_H1H1H3': c_H1H1H3,
                'c_H1H2H2': c_H1H2H2,
                'c_H1H2H3': c_H1H2H3,
                'c_H1H3H3': c_H1H3H3,
                'c_H2H2H2': c_H2H2H2,
                'c_H2H2H3': c_H2H2H3,
                'c_H2H3H3': c_H2H3H3,
                'c_H3H3H3': c_H3H3H3,
                # H1 cxn
                'x_H1W': x_H1W,
                'x_H1Z': x_H1Z,
                'x_H1_bb': x_H1_bb,
                'x_H1_gg': x_H1_gg,
                'x_H1_vbf': x_H1_vbf,
                'x_H1tt' : x_H1tt,
                # H2 cxn
                'x_H2W' : x_H2W,
                'x_H2Z' : x_H2Z,
                'x_H2_bb': x_H2_bb,
                'x_H2_gg': x_H2_gg,
                'x_H2_vbf': x_H2_vbf,
                'x_H2_tt': x_H2tt,
                # H3 cxn
                'x_H3W': x_H3W,
                'x_H3Z': x_H3Z,
                'x_H3_bb': x_H3_bb,
                'x_H3_gg': x_H3_gg,
                'x_H3_vbf': x_H3_vbf,
                'x_H3tt': x_H3tt,
                #Largeur Total
                'Wh1': Wh1,
                'Wh2': Wh2,
                'Wh3': Wh3, 
                'hexp': reshb.selectedLimits['h1'].expRatio(),
                'hobs': reshb.selectedLimits['h1'].obsRatio(),
                'hcha': reshb.selectedLimits['h1'].limit().citeKey(),

                'aexp': reshb.selectedLimits['h2'].expRatio(),
                'aobs': reshb.selectedLimits['h2'].obsRatio(),
                'acha': reshb.selectedLimits['h2'].limit().citeKey(),

                'xexp': reshb.selectedLimits['h3'].expRatio(),
                'xobs': reshb.selectedLimits['h3'].obsRatio(),
                'xcha': reshb.selectedLimits['h3'].limit().citeKey()})
#            })
    if True :#reshb.selectedLimits['h1'].obsRatio()<=1.0 and reshb.selectedLimits['h2'].obsRatio() <= 1. and reshb.selectedLimits['h3'].obsRatio() <= 1.0:
        df = pd.DataFrame(data)
        df.to_csv(f'data_checked.csv')
        print('The number of allowed point', jj)

### Run HiggsTools on ScannerS datasets
type1_dataset = read_scanners_output('data-to-check.csv')
process_data(type1_dataset)





















