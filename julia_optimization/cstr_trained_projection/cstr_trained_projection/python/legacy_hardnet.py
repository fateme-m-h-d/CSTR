"""CSTR definition and export for the pinned KKT-HardNet 1.0.1 API."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
os.environ.setdefault('JAX_ENABLE_X64','True')
import argparse,csv,hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from kkthn import KKTHardNet
from kkthn.builder import Expression
from kkthn.backbone import make_batched_mlp_apply
jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
COMMIT='bc9ba8908e650659436f531724386508e6cf30e0'


def check_version():
    if importlib.metadata.version('kkt-hardnet')!='1.0.1':
        raise RuntimeError('Use the pinned older API in requirements_legacy_hardnet.txt, not the current main branch.')


def reference(x,cfg):
    t,a=np.asarray(x).T
    f=1e13*np.exp(-90000/(8.314*t));r=1e11*np.exp(-cfg['ear']/(8.314*t))
    lo,hi=np.zeros_like(a),np.minimum(a,1.)
    for _ in range(64):
        e=(lo+hi)/2;g=(1+30*r)*e-10*f*(a-e)*(2-2*e)**2
        hi=np.where(g>0,e,hi);lo=np.where(g<=0,e,lo)
    e=(lo+hi)/2
    return np.column_stack((a-e,2-2*e,3*e))


def physical_kkt(x,w,raw,lam,cfg):
    scales=np.asarray(cfg['output_scale']);ca,cb,cc=np.asarray(w)*scales
    t,a=x;f=1e13*np.exp(-90000/(8.314*t));r=1e11*np.exp(-cfg['ear']/(8.314*t))
    g=np.array([a-ca-10*f*ca*cb**2+10*r*cc,a+2-ca-cb-cc])
    J=np.array([[-1-10*f*cb**2,-20*f*ca*cb,10*r],[-1.,-1.,-1.]])*scales
    return np.r_[np.asarray(w)-np.asarray(raw)+J.T@np.asarray(lam),g]


def build_package(cfg,train=None):
    check_version()
    projection=dict(gn_tol=cfg['projection_tolerance'],gn_max_iters=cfg['projection_max_iterations'],
        gn_reg=cfg['gn_reg'],fb_eps=1e-8,newton_step_length=.5,armijo_alpha=1e-4,
        armijo_beta=.5,max_backtrack_iter=10,backward_reg=1e-8)
    model=KKTHardNet(name='CSTR_legacy_hardnet',train=train,projection=projection)
    x=model.add_parameter(['Temperature','Cao']);y=model.add_variable(['Ca','Cb','Cc'])
    t=x.Temperature*cfg['input_scale'][0];a=x.Cao*cfg['input_scale'][1]
    ca,cb,cc=[v*s for v,s in zip([y.Ca,y.Cb,y.Cc],cfg['output_scale'])]
    def exp(e):return Expression(lambda ctx,e=e:jnp.exp(e.eval(ctx)),f'exp({e.text})')
    f=1e13*exp(-90000/(8.314*t));r=1e11*exp(-cfg['ear']/(8.314*t))
    model.constraints.add(a-ca-10*f*ca*cb**2+10*r*cc==0,a+2-ca-cb-cc==0)
    return model


def export_model(model,directory,points,cfg):
    """No retraining: copy JAX affine weights, transposed, into torch Sequential."""
    import torch
    state=model._inference_state  # inspected, version-pinned 1.0.1 adapter
    params=state.params
    if isinstance(params,dict):raise ValueError('Expected the surrogate backbone, not inverse-mode parameters')
    layers=[];exported=[]
    for i,row in enumerate(params):
        W,b=np.asarray(row['W']),np.asarray(row['b'])
        layer=torch.nn.Linear(W.shape[0],W.shape[1],dtype=torch.float64)
        with torch.no_grad():layer.weight.copy_(torch.tensor(W.T));layer.bias.copy_(torch.tensor(b))
        layers.append(layer)
        exported.append(dict(kind='linear',weight=W.T.tolist(),bias=b.tolist()))
        if i+1<len(params):layers.append(torch.nn.ReLU());exported.append(dict(kind='relu'))
    backbone=torch.nn.Sequential(*layers).eval()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    np.savetxt(directory/'check_inputs.csv',points,delimiter=',',header='T,Cao',comments='')
    torch.save(backbone,directory/'backbone.pt')
    torch.save(backbone.state_dict(),directory/'backbone_state_dict.pt')
    (directory/'layers.json').write_text(json.dumps(exported))
    (directory/'config.json').write_text(json.dumps(cfg,indent=2))
    scales=np.asarray(cfg['input_scale']);yscale=np.asarray(cfg['output_scale'])
    xs=jnp.asarray(points/scales)
    raw=np.asarray(make_batched_mlp_apply()(params,xs))
    with torch.no_grad():torchraw=backbone(torch.tensor(points/scales,dtype=torch.float64)).numpy()
    mismatch=float(np.max(np.abs(raw-torchraw)))
    if mismatch>1e-10:raise RuntimeError(f'JAX/PyTorch backbone mismatch {mismatch}')
    # Package solver exposes iterations, full packed root and its KKT L2 norm.
    solve=jax.jit(jax.vmap(state.projection.solve_single,in_axes=(0,0)))
    iterations,z,residual=map(np.asarray,solve(xs,jnp.asarray(raw)))
    projected=z[:,:3];lam=z[:,3:5]
    public=np.asarray(model.predict(np.asarray(xs),projection_backend='jax'))
    public_mismatch=float(np.max(np.abs(public-projected)))
    independent=np.array([physical_kkt(x,w,h,l,cfg) for x,w,h,l in zip(points,projected,raw,lam)])
    full_norm=np.linalg.norm(independent,axis=1)
    good=np.isfinite(z).all(1)&np.isfinite(residual)&(residual<=cfg['projection_tolerance'])&(full_norm<=cfg['projection_tolerance']*1.01)
    np.savetxt(directory/'parity.csv',np.column_stack((points,raw*yscale,projected*yscale)),delimiter=',',comments='',
        header='T,Cao,raw_A,raw_B,raw_C,projected_A,projected_B,projected_C')
    np.savetxt(directory/'projection_diagnostics.csv',np.column_stack((lam,iterations,residual,full_norm,good)),delimiter=',',comments='',
        header='lambda_A,lambda_sum,iterations,package_KKT_L2,independent_KKT_L2,converged')
    metrics=dict(points=len(points),converged_points=int(good.sum()),all_converged=bool(good.all()),
        max_package_KKT_L2=float(np.max(residual)),max_independent_KKT_L2=float(np.max(full_norm)),
        max_iterations=int(np.max(iterations)),jax_torch_raw_max_difference=mismatch,
        package_solver_public_predict_max_difference=public_mismatch)
    (directory/'export_checks.json').write_text(json.dumps(metrics,indent=2))
    print('EXPORT CHECKS',metrics,flush=True)
    if public_mismatch>1e-10:raise RuntimeError('Package solve_single and public predict disagree')
    if not good.all():raise RuntimeError('Backbone exported, but some package projections failed tolerance. This export is NOT accepted for embedding parity; inspect projection_diagnostics.csv.')
    return metrics


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'artifacts_legacy/hardnet')
    p.add_argument('--data',type=Path,default=ROOT/'data/cstr.csv')
    p.add_argument('--epochs',type=int,default=1000);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--projection-tolerance',type=float,default=1e-8)
    p.add_argument('--projection-max-iterations',type=int,default=1000)
    p.add_argument('--gn-reg',type=float,default=1e-3)
    p.add_argument('--cons-alpha',type=float,default=.5)
    return p
