"""stage2/apps/music/hearing/calibrate.py - measured recognition thresholds from labeled takes."""
import numpy as np
from .gallery import descriptor_distance
from .trajectory import PLACES,STEPS,dtw_distance
__all__=["calibrate_threshold","calibrate_gallery","set_gallery_threshold"]
def _dist(a,b):
 ta=a.get("trajectory");tb=b.get("trajectory")
 if ta is not None and tb is not None:
  va=np.asarray(ta,dtype=np.float64);vb=np.asarray(tb,dtype=np.float64)
  if va.shape==(STEPS,PLACES) and vb.shape==(STEPS,PLACES): return float(dtw_distance(va,vb))
 return float(descriptor_distance(a["descriptor"],b["descriptor"]))
def calibrate_threshold(labeled_takes):
 takes=list(labeled_takes or []);same=[];diff=[]
 for i,a in enumerate(takes):
  for b in takes[i+1:]:
   d=_dist(a,b)
   (same if a.get("name")==b.get("name") else diff).append(float(d))
 same_max=max(same) if same else None;diff_min=min(diff) if diff else None;separated=bool(same_max is not None and diff_min is not None and same_max<diff_min)
 if same_max is not None and diff_min is not None: thr=0.5*(same_max+diff_min)
 elif same: thr=max(same)
 elif diff: thr=min(diff)
 else: thr=0.25
 return {"same_distances":same,"different_distances":diff,"same_max":same_max,"different_min":diff_min,"same_mean":float(np.mean(same)) if same else None,"different_mean":float(np.mean(diff)) if diff else None,"threshold":float(thr),"separated_cleanly":separated,"overlap":bool(same_max is not None and diff_min is not None and same_max>=diff_min),"same_count":len(same),"different_count":len(diff)}
def calibrate_gallery(gallery,labeled_takes):
 cal=calibrate_threshold(labeled_takes)
 if gallery is not None and hasattr(gallery,"set_threshold"): gallery.set_threshold(cal["threshold"])
 cal["gallery_threshold"]=float(gallery.threshold) if gallery is not None and hasattr(gallery,"threshold") else None;return cal
def set_gallery_threshold(gallery,calibration): return gallery.set_threshold(float(calibration["threshold"])) if hasattr(gallery,"set_threshold") else None
