"""Opt-in real-photo extraction comparison. Does not create labels or edit profiles.

python tests/compare_real_sampling.py PHOTO REPORT --output model_cache/verification
The tiled result reuses OLD centroids at a different scale: sensitivity experiment,
not a recommended profile conversion and not an accuracy evaluation.
"""
import argparse, base64, hashlib, io, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main

def run():
    parser=argparse.ArgumentParser();parser.add_argument('photo');parser.add_argument('report');parser.add_argument('--output',default='model_cache/verification')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report=json.loads(Path(args.report).read_text('utf-8'))
    raw=Path(args.photo).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==report['source_image_sha256'],'Photo/report mismatch'
    img=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    os.environ['HF_HOME']=str(main.ROOT/'model_cache');os.environ['HF_HUB_OFFLINE']='1'
    p=main.validated_profile(report['profile_id'])
    mask_path=out/(digest[:12]+'-foreground.png')
    if not mask_path.exists():mask_path.write_bytes(main.make_foreground_mask(img))
    mask=Image.open(mask_path).convert('L');summary={}
    for mode in ('standard','tiles_v1'):
        started=time.perf_counter()
        vec,crop,geo,sec=main.embeddings(img,tuple(report['roi_original_pixels']),report['parameters']['resolution'],mode)
        assert vec.shape[-1]==384 and np.isfinite(vec).all()
        assert str(next(main.MODEL.parameters()).device)=='cpu'
        result=main.classify(vec,crop,geo,p,sec,digest,mask,hashlib.sha256(mask_path.read_bytes()).hexdigest(),report['insulator_exclusions']['regions'])
        r=result['report'];r['diagnostic_only']='Old centroids at two scales; NO accuracy validation or profile conversion'
        (out/(mode+'-report.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        for field,name in (('overlay_base64','overlay'),('class_mask_base64','classes'),('ambiguity_mask_base64','ambiguous')):
            (out/(mode+'-'+name+'.png')).write_bytes(base64.b64decode(result[field]))
        summary[mode]={'shape':list(vec.shape),'device':'cpu','finite':True,'forward_seconds':sec,'total_seconds':round(time.perf_counter()-started,2),
                       'parameters':r['parameters'],'corrosion_pct':r['corrosion_pct_of_visible_metal'],
                       'ambiguous_metal_pct':r['low_similarity_margin_pct_visible_metal']}
        print(json.dumps({mode:summary[mode]}),flush=True)
    (out/'real-photo-comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':run()
