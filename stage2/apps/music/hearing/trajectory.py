"""stage2/apps/music/hearing/trajectory.py - 24x32 trajectory, DTW chi-square, pitch-invariant shifts +-4."""
import math, numpy as np
from .cochlea import PLACE_COUNT
__all__=["STEPS","PLACES","MAX_SHIFT","trim_span","make_trajectory","trajectory_from_frames","trajectory_with_span","shift_trajectory","chi_square_cost","dtw_distance","match_distance","compare_trajectories"]
STEPS=24;PLACES=PLACE_COUNT;MAX_SHIFT=4;_EPS=1e-12
def _e(f):
 if "energy" in f and f["energy"] is not None: return float(f["energy"])
 l=float(f.get("loud",f.get("loudness",-120.0)));return 0.0 if l<=-119 else float(10**(l/10))
def trim_span(frames):
 frames=list(frames or [])
 if not frames: return 0,0
 e=np.asarray([_e(f) for f in frames]);p=float(np.max(e)) if e.size else 0.0
 if p<=_EPS: return 0,0
 idx=np.where(e>=max(0.01*p,1e-9))[0];return (int(idx[0]),int(idx[-1])+1) if idx.size else (0,0)
def _resample(m,steps):
 m=np.asarray(m,dtype=np.float64)
 if m.size==0: return np.zeros((steps,PLACES))
 if m.shape[0]==1: return np.repeat(m,steps,axis=0)
 pos=np.linspace(0,m.shape[0]-1,steps);lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,m.shape[0]-1);fr=(pos-lo)[:,None];return m[lo]*(1-fr)+m[hi]*fr
def _norm(m):
 m=np.asarray(m,dtype=np.float64);n=np.linalg.norm(m,axis=1,keepdims=True);return np.divide(m,n,out=np.zeros_like(m),where=n>_EPS)
def trajectory_with_span(frames):
 frames=list(frames or []);s,e=trim_span(frames);tr=frames[s:e]
 if not tr: return np.zeros((STEPS,PLACES)),{"start_frame":s,"end_frame":e,"frame_count":0,"start_time":0.0,"end_time":0.0,"steps":STEPS,"places":PLACES}
 m=np.asarray([f["place"] for f in tr],dtype=np.float64)
 if m.shape[1]!=PLACES: raise ValueError("place vectors must have %d"%PLACES)
 return _norm(_resample(m,STEPS)),{"start_frame":int(s),"end_frame":int(e),"frame_count":len(tr),"start_time":float(tr[0].get("time",0)),"end_time":float(tr[-1].get("time",0)),"steps":STEPS,"places":PLACES}
def make_trajectory(frames): return trajectory_with_span(frames)[0]
trajectory_from_frames=make_trajectory
def shift_trajectory(t,shift):
 t=np.asarray(t,dtype=np.float64)
 if t.shape[1]!=PLACES: raise ValueError("trajectory places")
 shift=int(shift);o=np.zeros_like(t)
 if shift>0: o[:,shift:]=t[:,:-shift]
 elif shift<0: o[:,:shift]=t[:,-shift:]
 else: o[:]=t
 return o
def chi_square_cost(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape: raise ValueError("equal shapes required")
 d=a+b;mask=d>_EPS;return 0.0 if not np.any(mask) else float(0.5*np.sum(((a[mask]-b[mask])**2)/d[mask]))
def _as(t):
 v=np.asarray(t,dtype=np.float64)
 if v.shape!=(STEPS,PLACES): raise ValueError("trajectory shape (%d,%d)"%(STEPS,PLACES))
 return v
def dtw_distance(first,second):
 first=_as(first);second=_as(second);R,C=first.shape[0],second.shape[0];cost=np.array([[chi_square_cost(first[i],second[j]) for j in range(C)] for i in range(R)])
 acc=np.full((R,C),np.inf);ln=np.zeros((R,C),dtype=int);acc[0,0]=cost[0,0];ln[0,0]=1
 for i in range(R):
  for j in range(C):
   if i==0 and j==0: continue
   best=np.inf;bl=0
   for pi,pj in ((i-1,j),(i,j-1),(i-1,j-1)):
    if pi<0 or pj<0: continue
    if acc[pi,pj]<best: best=float(acc[pi,pj]);bl=int(ln[pi,pj])
   if math.isfinite(best): acc[i,j]=best+cost[i,j];ln[i,j]=bl+1
 return float("inf") if not math.isfinite(acc[-1,-1]) else float(acc[-1,-1]/max(1,int(ln[-1,-1])))
def compare_trajectories(first,second,pitch_invariant=False,max_shift=MAX_SHIFT):
 first=_as(first);second=_as(second)
 if not pitch_invariant: return {"distance":float(dtw_distance(first,second)),"shift":0,"pitch_invariant":False}
 max_shift=int(max_shift);best=None;bs=0
 for s in range(-max_shift,max_shift+1):
  d=float(dtw_distance(first,shift_trajectory(second,s)))
  if best is None or d<best: best=d;bs=s
 return {"distance":float(best),"shift":int(bs),"pitch_invariant":True,"max_shift":max_shift}
def match_distance(first,second,pitch_invariant=False,max_shift=MAX_SHIFT): return compare_trajectories(first,second,pitch_invariant,max_shift)["distance"]
