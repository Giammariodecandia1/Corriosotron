import unittest, tempfile, json, base64, io
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main

class PrototypeTests(unittest.TestCase):
    def test_region(self):
        im=Image.new("RGB",(800,600))
        self.assertEqual(main.region(im,"[0,0,1,1]"),(0,0,800,600))
        with self.assertRaises(Exception):main.region(im,"[.5,.5,.1,.9]")
    def test_train_and_analyze_without_weights(self):
        img=Image.new("RGB",(800,600))
        box=(0,0,800,600)
        geo={"box":box,"left":0,"top":49,"rw":392,"rh":294,"side":392,"source_width":800,"source_height":600}
        vec=np.zeros((28,28,384),np.float32)
        vec[:,:8,0]=1;vec[:,8:18,1]=1;vec[:,18:,2]=1
        pts=[{"x":.1,"y":.5,"label":"background"},{"x":.45,"y":.5,"label":"healthy"},{"x":.9,"y":.5,"label":"rust3"}]
        p=main.train(vec,img,geo,pts)
        self.assertIn("rust3",p["prototypes"])
        out=main.classify(vec,img,geo,p,0.1)
        self.assertTrue(0 < out["report"]["corrosion_pct_of_visible_metal"] < 100)
        self.assertEqual(out["report"]["relative_visual_intensity_1_to_3"],3.0)
        self.assertGreater(len(out["overlay_base64"]),100)
        report=out["report"]
        self.assertEqual(report["class_legend"],{"0":"background","1":"healthy","2":"rust1","3":"rust2","4":"rust3"})
        self.assertEqual(sum(report["class_pixel_counts"].values()),report["mask_dimensions_px"][0]*report["mask_dimensions_px"][1])
        mask=np.asarray(Image.open(io.BytesIO(base64.b64decode(out["class_mask_base64"]))))
        self.assertEqual({k:int((mask==int(i)).sum()) for i,k in report["class_legend"].items()},report["class_pixel_counts"])
        overlay=np.asarray(Image.open(io.BytesIO(base64.b64decode(out["overlay_base64"]))))
        self.assertGreater(len(np.unique(overlay.reshape(-1,3),axis=0)),1)
        pixels=mask.size;counts=report["class_pixel_counts"]
        metal=sum(counts[k] for k in main.LABELS[1:])
        rust=sum(counts[k] for k in main.LABELS[2:])
        self.assertEqual(report["visible_metal_pct_of_roi"],round(100*metal/pixels,2))
        self.assertEqual(report["corrosion_pct_of_visible_metal"],round(100*rust/metal,2))
        self.assertEqual(report["class_pct_of_visible_metal"]["rust3"],round(100*counts["rust3"]/metal,2))
        self.assertEqual(report["parameters"]["roi_fraction_of_source"],[0.0,0.0,1.0,1.0])
        self.assertEqual(out["class_id_labels"],list(main.LABELS))
        self.assertIn("class_id,classe,pixel",out["report_csv"])
    def _calibration_data(self):
        img=Image.new("RGB",(800,600));geo={"box":(0,0,800,600),"left":0,"top":49,"rw":392,"rh":294,"side":392,"source_width":800,"source_height":600}
        vec=np.zeros((28,28,384),np.float32);vec[:,:8,0]=1;vec[:,8:18,1]=1;vec[:,18:,2]=1
        return img,geo,vec
    def test_duplicate_same_class_patch_counts_once(self):
        img,geo,vec=self._calibration_data()
        pts=[{"x":.1,"y":.5,"label":"background"},{"x":.1,"y":.5,"label":"background"},
             {"x":.45,"y":.5,"label":"healthy"},{"x":.9,"y":.5,"label":"rust3"}]
        p=main.train(vec,img,geo,pts)
        self.assertEqual(p["sample_counts"]["background"],1)
    def test_conflicting_classes_in_same_patch_rejected(self):
        img,geo,vec=self._calibration_data()
        pts=[{"x":.1,"y":.5,"label":"background"},{"x":.1,"y":.5,"label":"healthy"}]
        with self.assertRaisesRegex(main.HTTPException,"Conflitto"):
            main.train(vec,img,geo,pts)
    def test_invalid_and_outside_roi_points_rejected(self):
        img,geo,vec=self._calibration_data()
        for point in ({"x":.2,"y":.5,"label":"not-a-class"},{"x":.9,"y":.5,"label":"background"}):
            with self.subTest(point=point),self.assertRaises(main.HTTPException):
                main.train(vec,img,geo,[point])
    def test_bottom_right_coordinate_maps_to_last_patch(self):
        img,geo,vec=self._calibration_data()
        self.assertEqual(main.patch_from_point({"x":1,"y":1,"label":"rust3"},img,geo,28),(24,27))
    def test_no_metal_export_is_explicit(self):
        img,geo,vec=self._calibration_data();vec[:]=0;vec[:,:,0]=1
        p={"prototypes":{"background":[1]+[0]*383,"healthy":[0,1]+[0]*382,"rust3":[0,0,1]+[0]*381},"profile_id":"test","schema_version":1,"checkpoint_revision":"rev"}
        out=main.classify(vec,img,geo,p,.1);r=out["report"]
        self.assertIsNone(r["corrosion_pct_of_visible_metal"])
        self.assertEqual(r["class_pixel_counts"]["healthy"],0)
        self.assertIsNone(r["class_pct_of_visible_metal"]["rust3"])
        mask=np.asarray(Image.open(io.BytesIO(base64.b64decode(out["class_mask_base64"]))))
        self.assertEqual(set(np.unique(mask)),{0})
    def test_external_mask_limits_counts_to_visible_roi(self):
        img,geo,vec=self._calibration_data()
        p=main.train(vec,img,geo,[{"x":.1,"y":.5,"label":"background"},
                                  {"x":.45,"y":.5,"label":"healthy"},
                                  {"x":.9,"y":.5,"label":"rust3"}])
        support=Image.new("L",img.size,0)
        support.paste(255,(0,0,400,600))
        out=main.classify(vec,img,geo,p,.1,foreground_mask=support,foreground_sha256="mask-sha")
        report=out["report"]
        exported=np.asarray(Image.open(io.BytesIO(base64.b64decode(out["class_mask_base64"]))))
        self.assertEqual(exported.shape,(600,800))
        self.assertTrue(np.all(exported[:,400:]==0))
        self.assertEqual(report["foreground_mask"]["foreground_pixels_in_roi"],240000)
        self.assertEqual(report["foreground_mask"]["foreground_pct_of_roi"],50.0)
        self.assertEqual(report["foreground_mask"]["sha256"],"mask-sha")
        counts=report["class_pixel_counts"]
        metal=sum(counts[k] for k in main.LABELS[1:])
        rust=sum(counts[k] for k in main.LABELS[2:])
        self.assertEqual(sum(counts.values()),exported.size)
        self.assertEqual(report["visible_metal_pct_of_roi"],round(100*metal/exported.size,2))
        self.assertEqual(report["corrosion_pct_of_visible_metal"],round(100*rust/metal,2) if metal else None)
        self.assertIn("foreground_mask_sha256,mask-sha",out["report_csv"])
    def test_external_mask_upload_rejects_mismatch_and_rgb(self):
        from fastapi import UploadFile
        img=Image.new("RGB",(800,600))
        def run(coroutine):
            try:coroutine.send(None)
            except StopIteration as done:return done.value
            self.fail("La lettura della maschera in memoria non deve sospendersi")
        def uploaded(mode,size):
            buffer=io.BytesIO();Image.new(mode,size).save(buffer,format="PNG")
            spool=tempfile.SpooledTemporaryFile(max_size=1024*1024)
            spool.write(buffer.getvalue());spool.seek(0)
            return UploadFile(file=spool,filename="mask.png")
        def check(mode,size):
            item=uploaded(mode,size)
            try:return run(main.foreground_from_upload(item,img))
            finally:item.file.close()
        with self.assertRaisesRegex(main.HTTPException,"stesse dimensioni"):
            check("L",(799,600))
        with self.assertRaisesRegex(main.HTTPException,"maschera grigia"):
            check("RGB",(800,600))
        mask,digest=check("L",(800,600))
        self.assertEqual(mask.size,img.size)
        self.assertEqual(len(digest),64)
    def test_profile_rejects_null_and_legacy_vectors_explicitly(self):
        with tempfile.TemporaryDirectory(dir=main.ROOT) as d:
            path=Path(d)/"profile.json"
            profile={"schema_version":1,"model":main.MODEL_ID,"checkpoint_revision":"rev",
                     "preprocessing":main.preprocessing_config(),"profile_id":"test",
                     "prototypes":{"background":[0]*384,"healthy":[0,1]+[0]*382,"rust3":[0,0,1]+[0]*381}}
            path.write_text(json.dumps(profile),encoding="utf-8")
            with patch.object(main,"PROFILE",path),patch.object(main,"model",return_value=(None,None)),patch.object(main,"CHECKPOINT_REVISION","rev"):
                with self.assertRaisesRegex(main.HTTPException,"nullo"):
                    main.validated_profile()
                path.write_text(json.dumps({"model":main.MODEL_ID,"prototypes":{}}),encoding="utf-8")
                with self.assertRaisesRegex(main.HTTPException,"precedente"):
                    main.validated_profile()
if __name__=="__main__":unittest.main()
