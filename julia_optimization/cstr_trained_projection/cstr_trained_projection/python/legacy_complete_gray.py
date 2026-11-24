"""Complete old-package NN + projection oracle, with implicit KKT derivatives.

Values use the package's tolerance-stopped solver and original initialization.
Derivatives are of its local converged KKT branch, not a finite unrolled trace.
No weights, projection metric, stopping settings or root initialization change.
"""
from functools import lru_cache
from pathlib import Path
import contextlib,json,sys
import numpy as np
import jax
import jax.numpy as jnp
from legacy_hardnet import check_version,KKTHardNet,make_batched_mlp_apply
jax.config.update('jax_enable_x64',True)

class CompleteGray:
    def __init__(self,directory):
        check_version()
        self.directory=Path(directory).resolve()
        self.cfg=json.loads((self.directory/'config.json').read_text())
        if self.cfg['method']!='hardnet_legacy_package':raise ValueError('Requires the legacy package checkpoint')
        with contextlib.redirect_stdout(sys.stderr):
            self.model=KKTHardNet().load(str(self.directory/self.cfg['package_metadata_relative']),verbose=False)
        self.state=self.model._inference_state
        self.scales=np.asarray(self.cfg['output_scale'],dtype=float)
        self.xscale=jnp.asarray(self.cfg['input_scale'],dtype=jnp.float64)
        self.apply=make_batched_mlp_apply()
        self.last_x=None;self.cache={}
        self.solve=jax.jit(self.state.projection.solve_single)
        eq=self.state.projection.eq_constraints
        params=self.state.params;apply=self.apply;xscale=self.xscale
        def residual(q):
            xs=q[:2]/xscale;w=q[2:5];lam=q[5:7]
            raw=apply(params,xs[None,:])[0]
            J=jax.jacfwd(eq,argnums=0)(w,xs)
            return jnp.concatenate((w-raw+J.T@lam,eq(w,xs)))
        self.residual=jax.jit(residual)
        self.first=jax.jit(jax.jacfwd(residual))
        self.second=jax.jit(jax.jacfwd(jax.jacfwd(residual)))

    def _evaluate(self,values,order=0):
        x=np.asarray(values,dtype=float).reshape(2)
        if not np.isfinite(x).all():raise ValueError('Nonfinite operating input')
        if self.last_x is None or not np.array_equal(x,self.last_x):
            xs=jnp.asarray(x)/self.xscale
            raw=self.apply(self.state.params,xs[None,:])[0]
            steps,z,error=self.solve(xs,raw)
            z=np.asarray(z,dtype=float);error=float(error)
            if not np.isfinite(z).all() or not np.isfinite(error) or error>self.cfg['projection_tolerance']:
                raise RuntimeError(f'Package projection failed at {x.tolist()}: KKT L2={error}, iterations={int(steps)}')
            self.last_x=x.copy()
            self.cache=dict(z=z,q=np.r_[x,z],value=self.scales*z[:3],residual=error,iterations=int(steps))
        c=self.cache
        if order>=1 and 'jacobian' not in c:
            Fq=np.asarray(self.first(jnp.asarray(c['q'])))
            A=Fq[:,2:];Zx=np.linalg.solve(A,-Fq[:,:2])
            if not np.isfinite(Zx).all():raise RuntimeError('Invalid implicit KKT derivative')
            c.update(A=A,Zx=Zx,jacobian=self.scales[:,None]*Zx[:3],condition=float(np.linalg.cond(A)))
        if order>=2 and 'hessian' not in c:
            Fqq=np.asarray(self.second(jnp.asarray(c['q'])))
            V=np.vstack((np.eye(2),c['Zx']))
            rhs=np.einsum('ia,kij,jb->kab',V,Fqq,V)
            Zxx=np.linalg.solve(c['A'],-rhs.reshape(5,4)).reshape(5,2,2)
            H=self.scales[:,None,None]*Zxx[:3]
            if not np.isfinite(H).all():raise RuntimeError('Invalid implicit KKT second derivative')
            c['hessian']=(H+H.transpose(0,2,1))/2
        return c

    def value(self,x):return self._evaluate(x)['value'].tolist()
    def jacobian_flat(self,x):return self._evaluate(x,1)['jacobian'].T.ravel().tolist()
    def weighted_hessian(self,x,weights):
        H=np.einsum('k,kij->ij',np.asarray(weights,dtype=float),self._evaluate(x,2)['hessian'])
        return [float(H[0,0]),float(H[1,0]),float(H[1,1])]
    def diagnostics(self,x):
        c=self._evaluate(x,1)
        return dict(KKT_L2=c['residual'],iterations=c['iterations'],KKT_condition=c['condition'])
    def independent_value(self,x):
        xs=np.asarray(x,dtype=float)/np.asarray(self.cfg['input_scale'])
        return (np.asarray(self.model.predict(xs,projection_backend='jax'))*self.scales).tolist()

@lru_cache(maxsize=8)
def load(directory):return CompleteGray(directory)
