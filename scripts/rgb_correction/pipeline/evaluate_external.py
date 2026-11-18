"""Apply frozen source regressors to the original A outputs and fixed datasets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import h5py
import lpips
import numpy as np
import piq
import torch
from freqdec.evaluate import ssim_per_image
from freqdec.extract_features_v3 import richer_features
from freqdec.model import FixedBandResidualNet, load_initial_state

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--psnr-only', action='store_true')
parser.add_argument('--population', action='append')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--seeds', nargs='+', type=int, default=[56, 57, 58])
args = parser.parse_args()
root = Path(__file__).resolve().parent
populations = json.loads((root / 'manifest.json').read_text())['populations']
ids = {'ccnoise_15', 'polyu_30_scenes', 'nind_reserve2_confirmation',
       'nind_reserve1_independent', 'sidd_cc_benchmark_29_scenes'}
if args.population:
    assert set(args.population) <= ids
    ids = set(args.population)
if args.smoke:
    ids = {'ccnoise_15'}
    args.seeds = [56]
w = torch.tensor([.299, .587, .114], device='cuda')
def project(x):
    return x - ((x @ w) / (w @ w))[:, None] * w
def load_references(seed):
    result = {}
    for suffix in ('', '_sidd_benchmark'):
        item = json.loads((root / f'seed{seed}_v3_common_median_rgb{suffix}.json').read_text())
        result.update({p['population']:p['records'] for p in item['populations']})
    return result
def read_image(handle, key, start, end):
    return torch.tensor(handle[key][start:end], dtype=torch.float32, device='cuda').permute(0, 3, 1, 2) / 255
def metric(image, clean):
    psnr = -10 * (image-clean).square().mean((1,2,3)).log10()
    ssim = ssim_per_image(image, clean)
    if args.smoke or args.psnr_only:
        return torch.stack((psnr, ssim, torch.zeros_like(psnr), torch.zeros_like(psnr)), dim=1)
    return torch.stack((psnr, ssim, perceptual(image*2-1, clean*2-1).flatten(),
                        texture(image, clean).flatten()), dim=1)
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

for pop in populations:
    if pop['id'] in ids:
        assert digest(root / 'data' / pop['filename']) == pop['sha256'], pop['id']
perceptual = None if (args.smoke or args.psnr_only) else lpips.LPIPS(net='alex', pretrained=True, verbose=False).cuda().eval()
texture = None if (args.smoke or args.psnr_only) else piq.DISTS(reduction='none').cuda().eval()
for seed in args.seeds:
    references = load_references(seed)
    pre = np.load(root / f'preprocessing_{seed}.npz')
    mlp = np.load(root / f'mlp_{seed}.npz')
    model = FixedBandResidualNet('A').cuda().eval()
    load_initial_state(model,root / f'seed{seed}_A_endpoint.pt')
    for pop in populations:
        if pop['id'] not in ids:
            continue
        groups = defaultdict(list)
        rows = []
        baseline = references[pop['id']]
        with h5py.File(root / 'data' / pop['filename']) as h, torch.inference_mode():
            assert len(baseline) == len(h['noisy']) == len(pop['entries'])
            for start in range(0,len(baseline),args.batch_size):
                end = start + args.batch_size
                noisy, clean = [read_image(h,k,start,end) for k in ('noisy','gt')]
                base = model(noisy*2-1)*.5+.5
                features, names = richer_features(noisy,base)
                assert list(pre['features']) == list(names)
                x = np.clip((features.cpu().numpy()-pre['mu'])/pre['sd'],-10,10)
                values = {
                    'original': np.array([r['effective_offset_rgb'] for r in baseline[start:end]]),
                    'ridge': np.column_stack((x,np.ones(len(x)))) @ pre['ridge'],
                    'mlp': (np.tanh(x @ mlp['arr_0']+mlp['arr_1']) @ mlp['arr_2']+mlp['arr_3'])*pre['scale'],
                }
                target = project((2*(clean-base)).mean((-2,-1)))
                offsets = {k:project(torch.as_tensor(v,device='cuda',dtype=torch.float32)) for k,v in values.items()}
                offsets['oracle'] = target
                raw = metric(base.clamp(0,1),clean)
                prior_raw = torch.tensor([r['base_psnr'] for r in baseline[start:end]],device='cuda')
                assert (raw[:,0]-prior_raw).abs().max() < .002, (seed,pop['id'],start)
                arms = {}
                for k, delta in offsets.items():
                    scores = (metric((base+delta[:,:,None,None]*.5).clamp(0,1),clean)-raw).cpu().numpy()
                    cos = torch.nn.functional.cosine_similarity(delta,target).cpu().numpy()
                    rmse = ((delta-target).square().mean(1).sqrt()*.5).cpu().numpy()
                    arms[k] = np.column_stack((scores,cos,rmse))
                for j, entry in enumerate(pop['entries'][start:end]):
                    assert baseline[start+j]['row'] == entry['row'] == start+j
                    unit = str(entry[pop['primary_unit']])
                    record = {k: v[j].tolist() for k,v in arms.items()}
                    groups[unit].append(record)
                    rows.append({'row':start+j,'unit':unit,'arms':record})
        metrics = ['psnr','ssim','lpips','dists','cosine','per_row_offset_rmse']
        if args.psnr_only:
            for row in rows:
                row['arms'].update({k: v[:2]+v[4:] for k,v in row['arms'].items()})
            metrics = metrics[:2]+metrics[4:]
        summary = {arm:dict(zip(metrics,np.mean([np.mean([r[arm] for r in records],axis=0)
                    for records in groups.values()],axis=0).tolist())) for arm in arms}
        payload = {'seed':seed,'population':pop['id'],'metrics':metrics,'units':len(groups),'summary':summary,'rows':rows}
        prefix = 'preview' if args.psnr_only else 'external'
        destination = root / f'{prefix}_{seed}_{pop["id"]}.json'
        if args.smoke:
            print(seed,pop['id'],{k:v['psnr'] for k,v in summary.items()},flush=True)
            continue
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,indent=2)+'\n')
        temporary.replace(destination)
        print(seed,pop['id'],summary,flush=True)
