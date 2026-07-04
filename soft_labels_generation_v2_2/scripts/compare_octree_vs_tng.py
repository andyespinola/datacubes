import json, sys, numpy as np, requests, time
from pathlib import Path
sys.path.insert(0,"/home/andy/pythonProjects/datacubes/soft_labels_generation_v2_2/src")
from aperturenet_labels.io import tng_reader, units, ssp_grid
from aperturenet_labels.phase_a import extractor, classifier

KEY=[l.split("=",1)[1].strip() for l in open("/home/andy/pythonProjects/datacubes/data/.env") if l.startswith("TNG_API_KEY")][0]
SNAP, SUB = 91, 571097
GID=f"TNG50-{SNAP}-{SUB}"
D=Path("/media/andy/Data/tng/snap91_validation"); D.mkdir(parents=True, exist_ok=True)
SIM="TNG50-1"
STAR="Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity,Potential,ParticleIDs"
GAS="Coordinates,Velocities,Masses,StarFormationRate,Density,InternalEnergy,ElectronAbundance,GFM_Metallicity"

cut=D/f"{GID}.cutout.hdf5"; ph2=D/f"{GID}.cutout_phase2.hdf5"; meta=D/f"{GID}.subhalo.json"
def dl():
    if not cut.exists():
        tng_reader.download_cutout_fields(SNAP,SUB,cut,KEY,query=f"stars={STAR}&gas={GAS}",simulation=SIM)
    if not ph2.exists():
        tng_reader.download_cutout_fields(SNAP,SUB,ph2,KEY,query="dm=Coordinates",simulation=SIM)
    if not meta.exists():
        r=requests.get(f"https://www.tng-project.org/api/{SIM}/snapshots/{SNAP}/subhalos/{SUB}/",
                       headers={"API-Key":KEY}, timeout=120)
        meta.write_text(json.dumps(r.json(), indent=2))
t=time.time(); dl(); print(f"assets descargados en {time.time()-t:.0f}s")
print(f"  cutout {cut.stat().st_size/1e6:.0f}MB  phase2 {ph2.stat().st_size/1e6:.0f}MB")

grid=ssp_grid.load_ssp_grid(Path("/home/andy/pythonProjects/datacubes/kinematic_moments/templates/MaStar_CB19.slog_1_5.fits.gz"))

def run(method):
    truth=tng_reader.load_cutout_truth(cut, meta, ph2)
    truth=units.convert_truth_units(truth)
    has_pot = truth.stellar_potential is not None
    out=D/f"feat_{method}.h5"
    extractor.run_extractor(truth, GID, out, grid, config=extractor.ExtractorConfig(potential_method=method))
    f=extractor.load_particle_features(out)
    cl=D/f"lab_{method}.h5"
    classifier.run_classifier(f, cl)
    import h5py
    with h5py.File(cl) as h: P=h["P_class"][()]
    return f["epsilon"], f["R"], P, has_pot

print("\ncorriendo SNAPSHOT (potencial TNG)...")
eps_s, R_s, P_s, hp = run("snapshot")
print(f"  usó Potential de TNG: {hp}")
print("corriendo OCTREE (nuestro método)...")
eps_o, R_o, P_o, _ = run("octree")

n=min(len(eps_s),len(eps_o))
eps_s,eps_o=eps_s[:n],eps_o[:n]; R=R_s[:n]
ls,lo=P_s[:n].argmax(1),P_o[:n].argmax(1)
CL=["bulge","disk","halo"]
print(f"\n=== {GID}: {n:,} estrellas ===")
print(f"epsilon:  rho={np.corrcoef(eps_s,eps_o)[0,1]:.4f}  RMSE={np.sqrt(np.mean((eps_s-eps_o)**2)):.4f}  "
      f"mediana TNG={np.median(eps_s):+.3f} octree={np.median(eps_o):+.3f}")
print(f"etiquetas: acuerdo argmax={100*(ls==lo).mean():.1f}%")
print(f"  fracciones TNG   : "+" ".join(f"{c}={ (ls==k).mean():.3f}" for k,c in enumerate(CL)))
print(f"  fracciones octree: "+" ".join(f"{c}={ (lo==k).mean():.3f}" for k,c in enumerate(CL)))
# desacuerdos por clase
dis=ls!=lo
if dis.sum():
    print(f"  desacuerdos: {dis.sum()} ({100*dis.mean():.1f}%); "
          f"eps_med(desac) TNG={np.median(eps_s[dis]):+.2f} oct={np.median(eps_o[dis]):+.2f}")
