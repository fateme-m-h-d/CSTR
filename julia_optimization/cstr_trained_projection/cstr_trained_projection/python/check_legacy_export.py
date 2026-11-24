"""Independently recheck the exported .pt backbone against saved package weights."""
from legacy_hardnet import *
import torch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory.resolve();cfg=json.loads((d/'config.json').read_text())
    model=KKTHardNet().load(str(d/cfg['package_metadata_relative']),verbose=False)
    pt=torch.load(d/'backbone.pt',map_location='cpu',weights_only=False).double().eval()
    points=np.genfromtxt(d/'parity.csv',delimiter=',',skip_header=1)[:,:2]
    xs=points/np.asarray(cfg['input_scale'])
    native=np.asarray(make_batched_mlp_apply()(model._inference_state.params,jnp.asarray(xs)))
    with torch.no_grad():converted=pt(torch.tensor(xs,dtype=torch.float64)).numpy()
    diff=float(np.max(np.abs(native-converted)))
    print('JAX/package vs .pt raw scaled-output maximum difference =',diff)
    if diff>1e-10:raise RuntimeError('Backbone export parity failed')
    print(json.dumps(json.loads((d/'export_checks.json').read_text()),indent=2))
