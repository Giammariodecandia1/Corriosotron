"""Opt-in HTTP test with REAL DINO and synthetic pictures; no accuracy claims."""
import base64, csv, io, json, time, uuid
from pathlib import Path
import httpx
import numpy as np
from PIL import Image, ImageDraw

def picture(second=False):
    im=Image.new('RGB',(1000,1200),(35,55,40));d=ImageDraw.Draw(im)
    d.rectangle((90,80,420,1100),fill=(205,205,190))
    d.rectangle((570,80,910,1100),fill=(140,65,25))
    for y in range(100,1100,35):
        d.line((580,y,900,y+10),fill=(185,100+(10 if second else 0),42),width=5)
    if second:d.ellipse((440,400,550,500),fill=(22,90,50))
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

def validate(o):
    r=o['report'];mask=np.asarray(Image.open(io.BytesIO(base64.b64decode(o['class_mask_base64']))))
    overlay=Image.open(io.BytesIO(base64.b64decode(o['overlay_base64'])));overlay.load()
    ambiguity=np.asarray(Image.open(io.BytesIO(base64.b64decode(o['ambiguity_mask_base64']))))
    assert list(overlay.size)==r['mask_dimensions_px']
    assert {k:int((mask==int(i)).sum()) for i,k in r['class_legend'].items()}==r['class_pixel_counts']
    metal=int((mask>0).sum());rust=int((mask>=2).sum())
    assert r['corrosion_pct_of_visible_metal']==(round(100*rust/metal,2) if metal else None)
    assert int((ambiguity>0).sum())==r['ambiguous_metal_pixels']
    rows=list(csv.reader(io.StringIO(o['report_csv'])))
    assert sum(int(row[2]) for row in rows[2:])==mask.size
    assert r['device']=='CPU' and r['model']=='facebook/dinov2-small' and r['checkpoint_revision']
    assert r['profile_schema_version']==2 and r['parameters']['tile_count']>1
    return {k:r[k] for k in ('result_id','device','checkpoint_revision','profile_sample_counts','profile_source_images','inference_seconds','parameters')}

def run():
    root=Path(__file__).resolve().parents[1];out=root/'model_cache'/'verification';out.mkdir(exist_ok=True)
    points=[{'x':x,'y':y,'label':label} for x,label in ((.03,'background'),(.25,'healthy'),(.75,'rust1')) for y in (.2,.5,.8)]
    data={'seeds':json.dumps(points),'resolution':'518','feature_mode':'tiles_v1','profile_name':'TEST SOFTWARE '+uuid.uuid4().hex[:8]}
    summary={};pid=None
    with httpx.Client(base_url='http://127.0.0.1:8765',timeout=240) as client:
        try:
            files={'file':('synthetic-a.png',picture(),'image/png')}
            r=client.post('/api/calibrate',data=data,files=files);r.raise_for_status();o=r.json();pid=o['report']['profile_id']
            summary['calibrate']=validate(o)
            r=client.post('/api/analyze',data={'profile_id':pid},files={'file':('synthetic-b.png',picture(True),'image/png')});r.raise_for_status()
            summary['reuse_second_image']=validate(r.json())
            r=client.post(f'/api/profiles/{pid}/samples',data=data,files={'file':('synthetic-b.png',picture(True),'image/png')});r.raise_for_status()
            summary['append_second_image']=validate(r.json())
            r=client.post(f'/api/profiles/{pid}/samples',data=data,files=files)
            assert r.status_code==409,r.text;summary['duplicate_status']=r.status_code
            assert summary['append_second_image']['profile_source_images']==2
        finally:
            if pid:
                r=client.delete(f'/api/profiles/{pid}');r.raise_for_status()
                summary['test_profile_deleted']=True
    (out/'real-http.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
