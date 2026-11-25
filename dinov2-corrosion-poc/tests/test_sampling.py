"""Software tests with synthetic vectors: NOT corrosion accuracy tests."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import main
from feature_sampling import tile_plan, tiled_features, class_scores, merge_samples


class SamplingTests(unittest.TestCase):
    def test_tiling_coverage_and_coordinate_mapping(self):
        img=Image.new('RGB',(866,2807))
        gw,gh,side,xs,ys=tile_plan(*img.size,518)
        covered=np.zeros((gh,gw),bool)
        for y in ys:
            for x in xs:covered[y:y+side,x:x+side]=True
        self.assertTrue(covered.all())
        self.assertGreater(gw,37)
        def extract(canvas,box,size):
            a=np.zeros((size//14,size//14,384),np.float32);a[:,:,0]=1
            return a,canvas.crop(box),{},.1
        vec,crop,geo,_=tiled_features(img,(0,0,*img.size),518,extract)
        self.assertEqual(vec.shape,(gh,gw,384))
        np.testing.assert_allclose(vec[:,:,0],1)
        self.assertEqual(main.patch_from_point({'x':1,'y':1},img,geo,gh),(gh-1,gw-1))
        self.assertEqual(main.patch_from_point({'x':0,'y':0},img,geo,gh),(0,0))
        with self.assertRaises(ValueError):tile_plan(10000,10000,518)

    def test_multimodal_examples_not_collapsed_to_mean(self):
        # Three examples per visual mode. A single mean loses the two healthy modes.
        e=np.eye(384,dtype=np.float32)
        samples=[{'label':'healthy','embedding':v.tolist()} for v in [e[0]]*3+[e[1]]*3]
        samples += [{'label':'rust1','embedding':((e[0]+e[2])/np.sqrt(2)).tolist()}]*3
        p={'schema_version':2,'samples':samples,'prototypes':{'healthy':(e[0]+e[1]).tolist(),'rust1':(e[0]+e[2]).tolist()}}
        names,scores=class_scores(e[0].reshape(1,1,384),p,main.LABELS)
        self.assertEqual(names[scores[:,0,0].argmax()],'healthy')
        self.assertGreater(scores[0,0,0],scores[1,0,0])

    def test_named_profile_api_merge_and_ambiguity_export(self):
        e=np.eye(384,dtype=np.float32)
        def extract(img,box,size,mode='standard'):
            vec=np.zeros((28,28,384),np.float32)
            vec[:,:8]=e[0];vec[:,8:18]=e[1];vec[:,18:]=e[2]
            return vec,img.crop(box),{'left':0,'top':0,'rw':392,'rh':392,'side':392,'box':box,'source_width':img.width,'source_height':img.height},.01
        def photo(color):
            b=io.BytesIO();Image.new('RGB',(392,392),color).save(b,format='PNG');return b.getvalue()
        pts=[{'x':.1,'y':.5,'label':'background'},{'x':.45,'y':.5,'label':'healthy'},{'x':.9,'y':.5,'label':'rust1'}]
        with tempfile.TemporaryDirectory() as d,patch.object(main,'PROFILE_DIR',Path(d)),patch.object(main,'PROFILE',Path(d)/'absent'),patch.object(main,'model',return_value=(None,None)),patch.object(main,'CHECKPOINT_REVISION','test-revision'),patch.object(main,'embeddings',side_effect=extract):
            client=TestClient(main.app)
            data={'seeds':json.dumps(pts),'profile_name':'Prova','resolution':'392'}
            files={'file':('test.png',photo('white'),'image/png')}
            r=client.post('/api/calibrate',data=data,files=files)
            self.assertEqual(r.status_code,200,r.text)
            out=r.json();pid=out['report']['profile_id'];path=Path(d)/(pid+'.json')
            self.assertEqual(len(json.loads(path.read_text())['samples']),3)
            repeated=client.post(f'/api/profiles/{pid}/samples',data=data,files=files)
            self.assertEqual(repeated.status_code,409,repeated.text)
            files2={'file':('second.png',photo('silver'),'image/png')}
            added=client.post(f'/api/profiles/{pid}/samples',data=data,files=files2)
            self.assertEqual(added.status_code,200,added.text)
            profile=main.validated_profile(pid)
            self.assertEqual(len(profile['samples']),6)
            self.assertEqual(profile['sample_counts']['healthy'],2)
            self.assertEqual(added.json()['report']['profile_source_images'],2)
            analyze=client.post('/api/analyze',data={'profile_id':pid},files=files2)
            self.assertEqual(analyze.status_code,200,analyze.text)
            o=analyze.json();mask=np.asarray(Image.open(io.BytesIO(base64.b64decode(o['class_mask_base64']))))
            amb=np.asarray(Image.open(io.BytesIO(base64.b64decode(o['ambiguity_mask_base64']))))
            self.assertEqual(int((amb>0).sum()),o['report']['ambiguous_metal_pixels'])
            self.assertFalse(np.any((amb>0)&(mask==0)))
            conflict=[dict(pts[1],label='rust1')]
            bad=client.post(f'/api/profiles/{pid}/samples',data={'seeds':json.dumps(conflict)},files=files)
            self.assertEqual(bad.status_code,422,bad.text)
            self.assertEqual(len(main.validated_profile(pid)['samples']),6)
            client.delete(f'/api/profiles/{pid}')
            self.assertFalse(path.exists())

    def test_merge_does_not_mutate_on_conflict(self):
        profile={'samples':[{'sample_id':'a','label':'healthy'}]}
        with self.assertRaises(ValueError):merge_samples(profile,[{'sample_id':'b','label':'rust1'},{'sample_id':'a','label':'rust1'}])
        self.assertEqual(len(profile['samples']),1)

if __name__=='__main__':unittest.main()
