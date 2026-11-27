"""stage2/apps/music/hearing/segmentation.py - split recordings into utterances by frame energy."""
import numpy as np
__all__=["segment_frames","segment_utterances","split_recording"]
def segment_frames(frames,gap_limit_seconds=0.3,min_length_seconds=0.2,relative_threshold=0.05,energy_threshold=None):
 frames=list(frames or [])
 if not frames: return []
 energies=np.array([float(f.get("energy",0)) for f in frames])
 if energy_threshold is None: energy_threshold=float(np.max(energies))*float(relative_threshold) if energies.size else 0.0
 active=energies>=float(energy_threshold);spans=[];i=0
 while i<len(frames):
  if not active[i]: i+=1;continue
  j=i
  while j+1<len(frames) and active[j+1]: j+=1
  spans.append([i,j]);i=j+1
 merged=[]
 for s in spans:
  if merged and float(frames[s[0]].get("time",s[0]))-float(frames[merged[-1][1]].get("time",merged[-1][1]))<=float(gap_limit_seconds): merged[-1][1]=s[1]
  else: merged.append(list(s))
 out=[]
 for a,b in merged:
  start=float(frames[a].get("time",a));end=float(frames[b].get("time",b));dur=end-start
  if dur<float(min_length_seconds) and len(frames)>1: continue
  out.append({"start_frame":int(a),"end_frame":int(b),"start_time":start,"end_time":end,"duration":float(dur),"frame_count":int(b-a+1),"frames":frames[a:b+1]})
 return out
segment_utterances=segment_frames
def split_recording(frames,**kw): return segment_frames(frames,**kw)
