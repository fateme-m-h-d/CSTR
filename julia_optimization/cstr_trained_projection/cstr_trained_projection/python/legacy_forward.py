"""Package forward check at physical operating inputs; stdout is JSON only."""
from legacy_hardnet import *
import contextlib,sys

if __name__=='__main__':
    directory=Path(sys.argv[1]).resolve();x=np.array([[float(sys.argv[2]),float(sys.argv[3])]])
    cfg=json.loads((directory/'config.json').read_text())
    with contextlib.redirect_stdout(sys.stderr):
        model=KKTHardNet().load(str(directory/cfg['package_metadata_relative']),verbose=False)
        state=model._inference_state;xs=jnp.asarray(x/np.asarray(cfg['input_scale']))
        raw=make_batched_mlp_apply()(state.params,xs)
        iterations,z,residual=state.projection.solve_single(xs[0],raw[0])
        if not np.isfinite(np.asarray(z)).all() or float(residual)>cfg['projection_tolerance']:
            raise RuntimeError(f'Package projection at selected inputs did not converge: {float(residual)}')
        y=np.asarray(model.predict(np.asarray(xs),projection_backend='jax'))[0]*np.asarray(cfg['output_scale'])
    print(json.dumps(y.tolist()))
