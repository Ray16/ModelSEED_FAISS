"""Stock Boltz-2 on CUDA, with FP32 and the established serial-sampler fix."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import textwrap


def runtime_parameters():
    """Apply an explicit per-run memory policy without changing model quality."""
    settings=Path(__file__).resolve().parents[1]/'runtime_settings.json'
    if not settings.exists() or sys.argv[1:]==['--preflight']:return
    import yaml
    policy=json.loads(settings.read_text())
    if len(sys.argv)<3 or not Path(sys.argv[2]).is_file():return
    doc=yaml.safe_load(Path(sys.argv[2]).read_text())
    residues=sum(len(e['protein']['sequence']) for e in doc['sequences'] if 'protein' in e)
    parallel=int(policy['max_parallel_samples']) if residues<=int(policy['max_residues_for_parallel']) else 1
    if parallel<1 or parallel>4:raise ValueError('Unsupported memory policy')
    flag='--max_parallel_samples'
    if flag in sys.argv:sys.argv[sys.argv.index(flag)+1]=str(parallel)
    else:sys.argv.extend([flag,str(parallel)])
    record=dict(effective_argv=[sys.executable,*sys.argv],max_parallel_samples=parallel,
        protein_residues=residues,precision='FP32',sampling_steps_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_revision='cuda-fp32-memory-aware-batching-v2')
    if '--out_dir' in sys.argv:
        folder=Path(sys.argv[sys.argv.index('--out_dir')+1]).parent
        (folder/'effective_parameters.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EFFECTIVE_INFERENCE_PARAMETERS '+json.dumps(record,sort_keys=True),flush=True)


def patch():
    import torch
    import boltz.main as main
    from boltz.model.modules import diffusionv2
    source=textwrap.dedent(inspect.getsource(diffusionv2.AtomDiffusion.sample))
    pattern=r'sample_ids\.chunk\(\s*multiplicity % max_parallel_samples \+ 1\s*\)'
    if len(re.findall(pattern,source))!=1:raise RuntimeError('Unknown Boltz sampling implementation')
    fixed=re.sub(pattern,'sample_ids.split(max_parallel_samples)',source)
    namespace={};exec(compile(fixed,__file__,'exec'),diffusionv2.__dict__,namespace)
    diffusionv2.AtomDiffusion.sample=namespace['sample']
    original=main.Trainer
    def fp32_trainer(*args,**kwargs):
        kwargs['precision']=32
        return original(*args,**kwargs)
    main.Trainer=fp32_trainer
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS','8')))
    torch.set_num_interop_threads(1)
    metadata=dict(implementation='stock_boltz2_cuda_fp32_serial',torch=torch.__version__,
        precision='32-true',approximate_mode=False,sample_chunking='split(max_parallel_samples)',
        sampler_original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        boltz_main_sha256=hashlib.sha256(inspect.getsource(main).encode()).hexdigest(),
        changes='Process-local precision and established sampler correction; shared environment unchanged')
    print('LAMBDA_BOLTZ_IMPLEMENTATION '+json.dumps(metadata,sort_keys=True),flush=True)
    return main


if __name__=='__main__':
    runtime_parameters()
    main=patch()
    if sys.argv[1:]==['--preflight']:print('CUDA_FP32_WRAPPER_READY',flush=True)
    else:main.cli()
