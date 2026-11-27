"""stage2/apps/music/hearing/chladni.py - 78-mode square plate model, 55Hz ref, 55-14575Hz. Gaussian log-freq weighting, top modes, plate field, timeline."""
import math, numpy as np
from PIL import Image
from .cochlea import PLACE_COUNT, PLACE_FREQS
__all__=["REFERENCE_HZ","MIN_HZ","MAX_HZ","MODE_PAIRS","MODE_FREQS","MODES","MODE_TABLE","DEFAULT_SIGMA","mode_weights","weights_for_descriptor","top_modes","strongest_modes","plate_field","draw_plate","frame_weights","timeline_weights","timeline_figures"]
REFERENCE_HZ=55.0;MIN_HZ=55.0;MAX_HZ=14575.0
MODE_PAIRS=[(m,n) for m in range(1,13) for n in range(m+1,14)]
_RMIN=min(m*m+n*n for m,n in MODE_PAIRS);_RMAX=max(m*m+n*n for m,n in MODE_PAIRS)
def _mf(m,n):
 r=m*m+n*n;return MIN_HZ+((r-_RMIN)/(_RMAX-_RMIN))*(MAX_HZ-MIN_HZ) if _RMAX!=_RMIN else REFERENCE_HZ
MODE_FREQS=[_mf(m,n) for m,n in MODE_PAIRS]
MODES=[{"mode":i,"pair":[m,n],"m":m,"n":n,"frequency_hz":float(f),"freq_hz":float(f)} for i,((m,n),f) in enumerate(zip(MODE_PAIRS,MODE_FREQS))]
MODE_TABLE=MODES;DEFAULT_SIGMA=0.35;_EPS=1e-12
def _desc(d):
 v=np.asarray(d,dtype=np.float64)
 if v.size!=PLACE_COUNT: raise ValueError("descriptor must have %d places"%PLACE_COUNT)
 v=np.clip(np.nan_to_num(v,nan=0.0,posinf=0.0,neginf=0.0),0.0,None);t=float(np.sum(v));return v/t if t>_EPS else v
def mode_weights(descriptor,place_freqs=None,sigma=DEFAULT_SIGMA):
 if sigma is None or float(sigma)<=0: raise ValueError("sigma must be positive")
 sigma=float(sigma);pv=_desc(descriptor);fr=np.asarray(PLACE_FREQS if place_freqs is None else place_freqs,dtype=np.float64)
 if fr.size!=PLACE_COUNT: raise ValueError("place_freqs must have %d entries"%PLACE_COUNT)
 diff=np.log(np.asarray(MODE_FREQS))[:,None]-np.log(fr)[None,:];w=np.exp(-0.5*(diff/sigma)**2)@pv;t=float(np.sum(w));w=w/t if t>_EPS else np.zeros(len(MODE_FREQS))
 return [float(x) for x in w]
weights_for_descriptor=mode_weights
def top_modes(weights,count=5):
 v=[float(x) for x in weights]
 if len(v)!=len(MODES): raise ValueError("weights must have %d entries"%len(MODES))
 o=sorted(range(len(v)),key=lambda i:v[i],reverse=True);return [{"mode":int(i),"pair":list(MODES[i]["pair"]),"frequency_hz":float(MODES[i]["frequency_hz"]),"weight":float(v[i])} for i in o[:max(0,int(count))]]
strongest_modes=top_modes
def _field(m,n,s):
 a=(np.arange(s)+0.5)/s;return np.sin(math.pi*m*a[None,:])*np.sin(math.pi*n*a[:,None])
def plate_field(weights,size=128):
 v=np.asarray(weights,dtype=np.float64)
 if v.size!=len(MODES): raise ValueError("weights must have %d entries"%len(MODES))
 size=int(size)
 if size<=0: raise ValueError("size must be positive")
 f=np.zeros((size,size))
 for i in np.argsort(-v)[:12]:
  if v[i]>_EPS: m,n=MODE_PAIRS[int(i)];f+=float(v[i])*_field(m,n,size)
 p=float(np.max(np.abs(f))) if f.size else 0.0
 if p>_EPS: f=f/p
 return np.clip(0.5+0.5*f,0,1)
def draw_plate(weights,size=128): return Image.fromarray(np.clip(plate_field(weights,size)*255,0,255).astype(np.uint8),mode="L")
def frame_weights(frames,sigma=DEFAULT_SIGMA): return [mode_weights(f["place"],sigma=sigma) for f in frames]
timeline_weights=frame_weights
def timeline_figures(frames,sigma=DEFAULT_SIGMA,size=64,max_figures=None):
 w=frame_weights(frames,sigma=sigma)
 if max_figures is not None: w=w[:int(max_figures)]
 return [draw_plate(x,size=size) for x in w]
