"""Re-export a saved package model without retraining; requires this workflow's config."""
from legacy_hardnet import *

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    directory=args.directory.resolve();cfg=json.loads((directory/'config.json').read_text())
    model=KKTHardNet().load(str(directory/cfg['package_metadata_relative']),verbose=False)
    if (directory/'check_inputs.csv').exists():
        points=np.genfromtxt(directory/'check_inputs.csv',delimiter=',',skip_header=1)
    else:
        data=np.genfromtxt(ROOT/'data/cstr.csv',delimiter=',',skip_header=1)
        points=np.vstack((data[:,:2],[[391.41,1.0111],[382.78591974,1.2],[447.9235,.94952]]))
    export_model(model,directory,points,cfg)
