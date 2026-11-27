"""Assumption-only property scenarios. No training or target-based parameter fitting."""
from pathlib import Path
import csv, hashlib, itertools, json, math

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
HEAD = '74c8d23c70e7cc6276a8d65a2df7a978633b2822'
RHO_S = {'hexane':659., 'toluene':867., 'ethyl_acetate':902.}
DESCRIPTORS = {(3.9,.377,17.89):'hexane',(2.7,.680,29.46):'toluene',(.7,.423,24.):'ethyl_acetate'}
# Linear interpolation between manufacturer 50 and 100 cSt analogues at 65 cSt.
RHO_P = 960. + (65.-50.)/(100.-50.)*(966.-960.)
GAMMA_P = 20.8 + (65.-50.)/(100.-50.)*(20.9-20.8)
ETA_P = 65. * RHO_P / 1000.  # cSt * g/mL = cP; neat PDMS, not solution
PHYS = ['Withdrawal Velocity (mm/s)','Viscosity (cP)','Surface Tension (mN/m)',
        'Solution Density (kg/m3)','Wet-to-Dry Mobile Factor','LLD Applicable']
FILL = PHYS + ['measurement_source','unit_confirmation','regime_evidence']

def read(p):
    with p.open(newline='') as f:
        r=csv.DictReader(f); return r.fieldnames,list(r)

def write(p, rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def lld(eta,gamma,rho,u=1.):
    ca=eta*1e-3*u*1e-3/(gamma*1e-3)
    return .9458*math.sqrt(gamma*1e-3/(rho*9.81))*ca**(2/3)*1e9,ca

def estimate(c, eta_s, gamma_s, solvent):
    """Only recipe inputs; measured thickness labels are deliberately absent."""
    if not (math.isfinite(c) and 0 < c < RHO_P): return None
    phi=c/RHO_P  # g/L numerically equals kg/m3
    rho=RHO_S[solvent]*(1-phi)+RHO_P*phi
    eta=math.exp((1-phi)*math.log(eta_s)+phi*math.log(ETA_P))
    gamma=(1-phi)*gamma_s+phi*GAMMA_P
    wet,ca=lld(eta,gamma,rho)
    # Alternative mixture models and analogue endpoints. NOT confidence bounds.
    eta_candidates=[eta_s,eta,(1-phi)*eta_s+phi*ETA_P]
    gamma_candidates=[gamma_s,gamma,GAMMA_P]
    rho_candidates=[RHO_S[solvent]*(1-c/rp)+c for rp in (960.,966.)]
    phi_candidates=[c/rp for rp in (960.,966.)]
    wet_values=[lld(e,g,r)[0] for e,g,r in itertools.product(eta_candidates,gamma_candidates,rho_candidates)]
    return dict(phi=phi,rho=rho,eta=eta,gamma=gamma,wet=wet,ca=ca,
                eta_low=min(eta_candidates),eta_high=max(eta_candidates),
                gamma_low=min(gamma_candidates),gamma_high=max(gamma_candidates),
                rho_low=min(rho_candidates),rho_high=max(rho_candidates),
                wet_low=min(wet_values),wet_high=max(wet_values),
                dry_high=max(wet_values)*max(phi_candidates))

def main():
    manifest={'source_commit':HEAD,'date':'2026-11-27','datasets':[],
              'status':'assumption scenarios, not measured properties or validated tolerance intervals',
              'pdms_analogue':{'rho_kg_m3':RHO_P,'surface_tension_mN_m':GAMMA_P,'neat_viscosity_cP':ETA_P},
              'training_performed':False,'hydrodynamic_weight':0,'velocity_smooth_weight':0}
    summaries=[]
    for kind in ['experimental','synthetic']:
        source=BASE/f'pinn_{kind}_physics_template.csv'
        cols,rows=read(source); out=[]
        for i,row in enumerate(rows):
            r=dict(row)
            # Prevent regeneration from overwriting a subsequently filled intake.
            assert all(not row[k] for k in FILL), 'Source intake already contains values'
            if kind=='experimental':
                c=float(row['concentration_g_l']); solvent=row['solvent']
                eta_s=float(row['viscosity_cp']); gamma_s=float(row['surface_tension_mn_m'])
                total=float(row['total_nm']);bonded=float(row['bonded_nm'])
            else:
                c=float(row['Concentration (g/mL)'])
                eta_s=float(row['Legacy Descriptor Viscosity (cP)']);gamma_s=float(row['Legacy Descriptor Surface Tension (mN/m)'])
                solvent=DESCRIPTORS[(float(row['Polarity (XLogP3)']),eta_s,gamma_s)]
                total=float(row['Total Thickness (nm)']);bonded=float(row['Bonded Thickness (nm)'])
            x=estimate(c,eta_s,gamma_s,solvent)
            r.update({'Withdrawal Velocity (mm/s)':1.,'LLD Applicable':0,
                'measurement_source':'Fajer email supplied by Alex; other fields ESTIMATED: see ASSUMPTIONS.md',
                'unit_confirmation':'Physics concentration uses g/L = kg/m3; legacy source header unchanged',
                'regime_evidence':'UNVALIDATED; LLD training disabled; evaporation/adsorption unresolved',
                'physics_solvent':solvent,'physics_concentration_g_l':c,
                'physics_row_id':f'{kind}:{i+2}',
                'speed_status':'1 mm/s standard-protocol assumption; individual row linkage unverified',
                'Dwell Time (s)':20.,'dwell_status':'standard-protocol assumption; individual row linkage unverified',
                'property_status':'hypothetical mixture estimates; no solution measurements',
                'concentration_status':('canonical_g_l' if kind=='experimental' else 'inherited_g_l_interpretation_from_repository_audit; parentage_unavailable'),
                'temperature_status':'ambient reference scenario only; actual temperature unknown; no temperature correction',
                'wet_dry_status':'q=1 limiting entrainment-only scenario; NOT calibrated mobile conversion',
                'baseline_label_eligible':int(c>0 and total>0 and 0<=bonded<=total),
                'physics_estimate_valid':int(x is not None),
                'physics_issue':('nonpositive_or_impossible_concentration' if x is None else 'LLD_regime_and_mobile_partition_unvalidated'),
                'polymer_volume_fraction':'','viscosity_scenario_min_cp':'','viscosity_scenario_max_cp':'',
                'surface_tension_scenario_min_mn_m':'','surface_tension_scenario_max_mn_m':'',
                'density_analogue_min_kg_m3':'','density_analogue_max_kg_m3':'',
                'capillary_number_scenario':'','lld_wet_scenario_nm':'',
                'lld_wet_model_min_nm':'','lld_wet_model_max_nm':'',
                'mobile_q0_scenario_nm':'','mobile_q05_scenario_nm':'','mobile_q1_scenario_nm':'',
                'mobile_model_min_nm':'','mobile_model_max_nm':''})
            if x:
                r.update({'Viscosity (cP)':x['eta'],'Surface Tension (mN/m)':x['gamma'],
                    'Solution Density (kg/m3)':x['rho'],'Wet-to-Dry Mobile Factor':x['phi'],
                    'polymer_volume_fraction':x['phi'],'viscosity_scenario_min_cp':x['eta_low'],
                    'viscosity_scenario_max_cp':x['eta_high'],'surface_tension_scenario_min_mn_m':x['gamma_low'],
                    'surface_tension_scenario_max_mn_m':x['gamma_high'],'density_analogue_min_kg_m3':x['rho_low'],
                    'density_analogue_max_kg_m3':x['rho_high'],'capillary_number_scenario':x['ca'],
                    'lld_wet_scenario_nm':x['wet'],'lld_wet_model_min_nm':x['wet_low'],'lld_wet_model_max_nm':x['wet_high'],
                    'mobile_q0_scenario_nm':0.,'mobile_q05_scenario_nm':.5*x['phi']*x['wet'],
                    'mobile_q1_scenario_nm':x['phi']*x['wet'],'mobile_model_min_nm':0.,'mobile_model_max_nm':x['dry_high']})
            out.append(r)
        target=HERE/f'pinn_{kind}_physics_assumed.csv';write(target,out)
        _,verified=read(target)
        for old,new in zip(rows,verified):
            assert all(old[k]==new[k] for k in cols if k not in FILL), 'Original data changed'
            assert new['LLD Applicable']=='0'
        valid=[r for r in out if r['physics_estimate_valid']]
        manifest['datasets'].append({'kind':kind,'source':source.name,'source_sha256':sha(source),
             'output':target.name,'output_sha256':sha(target),'rows':len(rows),
             'property_rows_populated':len(valid),'property_rows_unresolved':len(out)-len(valid),
             'baseline_label_eligible':sum(r['baseline_label_eligible'] for r in out),
             'preservation_check':'all original non-intake cell strings and row order preserved'})
        for solvent in RHO_S:
            group=[r for r in valid if r['physics_solvent']==solvent]
            summaries.append({'dataset':kind,'solvent':solvent,'rows_with_estimates':len(group),
                **{key+suffix:fun(float(r[key]) for r in group) for key in ['physics_concentration_g_l','Viscosity (cP)',
                 'Surface Tension (mN/m)','Solution Density (kg/m3)','Wet-to-Dry Mobile Factor','mobile_q1_scenario_nm']
                 for suffix,fun in [('_min',min),('_max',max)]}})
    write(HERE/'property_summary.csv',summaries)
    # Analytic limit and dimensional checks independent of data preservation checks.
    x=estimate(1e-12,.377,17.89,'hexane')
    assert math.isclose(x['eta'],.377,rel_tol=1e-10)
    assert math.isclose(x['rho'],659.,rel_tol=1e-10)
    assert math.isclose(lld(.377,17.89,659.,2)[0]/lld(.377,17.89,659.,1)[0],2**(2/3))
    assert estimate(-1,.377,17.89,'hexane') is None
    assert math.isclose(65*RHO_P/1000,62.517)
    manifest['checks']=['source preservation','row counts','LLD disabled','zero-concentration solvent limit',
                        'LLD U^(2/3) scaling','negative concentration rejection','cSt to cP conversion']
    manifest['script_sha256']=sha(Path(__file__))
    (HERE/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
