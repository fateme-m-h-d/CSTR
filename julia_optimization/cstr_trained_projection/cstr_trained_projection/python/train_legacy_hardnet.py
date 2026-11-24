"""Actual package training with the older API, followed by backbone .pt export."""
from legacy_hardnet import *
import contextlib


def train(args):
    check_version()
    if args.output.exists():raise FileExistsError('Choose a new --output folder; do not overwrite earlier checkpoints')
    if not 1<=args.warmup<args.epochs:raise ValueError('Require 1 <= warmup < epochs')
    data=np.genfromtxt(args.data,delimiter=',',skip_header=1)
    if data.ndim!=2 or data.shape[1]!=5 or not np.isfinite(data).all():raise ValueError('Expected finite T,Cao,Ca,Cb,Cc data')
    # Physical parameters agree with julia/exact.jl and the existing optimization data.
    cfg=dict(method='hardnet_legacy_package',ear=100000.,tau=10.,R=8.314,
        Af=1e13,Eaf=90000.,Ar=1e11,CB0=2.,CC0=0.,source_commit=COMMIT,
        activation='relu',dtype='float64',projection_metric='scaled_euclidean',
        input_scale=np.max(np.abs(data[:,:2]),axis=0).tolist(),
        output_scale=np.max(np.abs(data[:,2:]),axis=0).tolist(),
        projection_tolerance=args.projection_tolerance,projection_max_iterations=args.projection_max_iterations,
        gn_reg=args.gn_reg,epochs=args.epochs,warmup=args.warmup,lr=args.lr,hidden=args.hidden,
        batch_size=args.batch_size,seed=args.seed,cons_alpha=args.cons_alpha,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest())
    if np.max(np.abs(reference(data[:,:2],cfg)-data[:,2:]))>1e-5:
        raise ValueError('Training data disagree with the optimization reference at Ear=100000; regenerate/reconcile the data first')
    idx=np.random.RandomState(42).permutation(len(data));n,m=int(.6*len(data)),int(.2*len(data))
    split=dict(train=idx[:n].tolist(),validation=idx[n:n+m].tolist(),test=idx[n+m:].tolist())
    pool=np.r_[idx[:n],idx[n:n+m]];scaled=data/np.r_[cfg['input_scale'],cfg['output_scale']]
    order=np.random.default_rng(args.seed).permutation(len(pool))
    if int(.75*len(pool))!=n:raise ValueError('Dataset size incompatible with the intended 60/20/20 split')
    arranged=np.empty((len(pool),5));arranged[order]=scaled[pool]
    directory=args.output.resolve();directory.mkdir(parents=True)
    (directory/'split.json').write_text(json.dumps(split,indent=2))
    np.savetxt(directory/'parameters.csv',arranged[:,:2],delimiter=',',header='Temperature,Cao',comments='')
    np.savetxt(directory/'variables.csv',arranged[:,2:],delimiter=',',header='Ca,Cb,Cc',comments='')
    train_cfg=dict(epochs=args.epochs,epoch_mlp=args.warmup,batch_size=args.batch_size,
        learning_rate=args.lr,train_frac=.75,hidden_size=args.hidden,hidden_layers=2,seed=args.seed,
        dtype='float64',print_every=100,drop_last=False,cons_alpha=args.cons_alpha)
    model=build_package(cfg,train_cfg)
    model.dataset(parameters=directory/'parameters.csv',variables=directory/'variables.csv')
    previous=Path.cwd()
    try:
        os.chdir(directory)
        result=model.model()  # train via the legacy PACKAGE; no custom training loop
    finally:os.chdir(previous)
    bundle=Path(result['metadata_path']).resolve()
    cfg['package_metadata_relative']=str(bundle.relative_to(directory))
    points=np.vstack((data[:,:2],[[391.41,1.0111],[382.78591974,1.2],[447.9235,.94952]]))
    # 1.0.1's training return leaves its cached predictor unset. Reload through
    # the public API, which initializes that cache, before testing predictions.
    model=KKTHardNet().load(str(bundle),verbose=False)
    export_model(model,directory,points,cfg)
    parity=np.genfromtxt(directory/'parity.csv',delimiter=',',skip_header=1)
    ids=split['test']
    predictive=dict(test_physical_rmse=float(np.sqrt(np.mean((parity[ids,5:]-data[ids,2:])**2))))
    (directory/'predictive_metrics.json').write_text(json.dumps(predictive,indent=2))
    print('Backbone for MathOptAI:',directory/'backbone.pt',flush=True)

if __name__=='__main__':train(parser().parse_args())
