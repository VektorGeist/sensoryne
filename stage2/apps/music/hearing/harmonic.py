"""stage2/apps/music/hearing/harmonic.py - harmonic stack scoring MIDI 21-108, octave guard, polyphony top5, disagreement flag."""
import math, numpy as np
from ..instruments import nearest_note
__all__=["score_midi","harmonic_pitch","harmonic_pitch_for_segment","polyphony_candidates"]
def _mag(samples,sample_rate):
 x=np.asarray(samples,dtype=np.float64)
 if x.ndim>1: x=x[:,0]
 if x.size and np.max(np.abs(x))>1.5: x=x/32768.0
 n=1
 while n<len(x)*4: n*=2
 win=np.hanning(len(x)) if len(x)>1 else np.ones_like(x);spec=np.fft.rfft(x*win,n=n);return np.abs(spec),float(sample_rate)/n
def _at(mag,bin_hz,f):
 if f<=0: return 0.0
 i=int(round(f/bin_hz));return float(mag[i]) if 0<=i<len(mag) else 0.0
def score_midi(samples,sample_rate=44100):
 mag,bin_hz=_mag(samples,sample_rate);sr=float(sample_rate);base={}
 for midi in range(21,109):
  f=440.0*(2**((midi-69)/12))
  if f>=sr/2: continue
  s=0.0
  for h,w in ((1,1.0),(2,0.5),(3,1/3),(4,0.25)):
   if h*f<sr/2: s+=w*_at(mag,bin_hz,h*f)
  base[midi]=float(s)
 guarded={}
 for midi,score in base.items():
  below=base.get(midi-12);guarded[midi]=float(score*0.5 if below is not None and below>=0.9*score and score>0 else score)
 return guarded
def harmonic_pitch(samples,sample_rate=44100,cochlea_f0=None):
 scores=score_midi(samples,sample_rate)
 if not scores: return {"harmonic_pitch_hz":0.0,"harmonic_midi":None,"confidence":0.0,"candidates":[],"disagreement":False,"disagreement_flag":False,"cochlea_f0":cochlea_f0}
 best_midi=max(scores,key=lambda m:scores[m]);best=float(scores[best_midi]);ordered=sorted(scores.values(),reverse=True);second=float(ordered[1]) if len(ordered)>1 else 0.0
 conf=float((best-second)/best) if best>0 else 0.0;freq=440.0*(2**((best_midi-69)/12))
 cand=[]
 for midi in sorted(scores,key=lambda m:scores[m],reverse=True):
  if any(abs(midi-c["midi"])<=1 for c in cand): continue
  f=440.0*(2**((midi-69)/12));cand.append({"midi":int(midi),"note":nearest_note(f)["name"],"freq_hz":float(f),"score":float(scores[midi]),"normalized_score":float(scores[midi]/best) if best>0 else 0.0})
  if len(cand)>=5: break
 disagreement=False
 if cochlea_f0 and float(cochlea_f0)>0 and freq>0: disagreement=bool(abs(math.log2(freq/float(cochlea_f0)))>1.0)
 return {"harmonic_pitch_hz":float(freq),"harmonic_midi":int(best_midi),"note":nearest_note(freq)["name"],"confidence":float(conf),"candidates":cand,"polyphony_candidates":cand,"disagreement":disagreement,"disagreement_flag":disagreement,"cochlea_f0":float(cochlea_f0) if cochlea_f0 else None,"scores":{str(k):float(v) for k,v in scores.items()}}
def harmonic_pitch_for_segment(samples,sample_rate=44100,cochlea_f0=None): return harmonic_pitch(samples,sample_rate,cochlea_f0)
def polyphony_candidates(samples,sample_rate=44100): return harmonic_pitch(samples,sample_rate)["candidates"]
