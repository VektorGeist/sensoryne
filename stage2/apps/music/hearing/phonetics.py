"""stage2/apps/music/hearing/phonetics.py - manner fractions, formants F1-F3, VOT, phonetic/word sentences. Measures only."""
import math, numpy as np
__all__=["manner_features","manner_fractions","track_formants","formant_transitions","voice_onset_time","phonetic_summary","word_reading_sentence","analyze_phonetics"]
_EPS=1e-12
def manner_features(frames):
 n=len(frames)
 if not n: return {"voiced":0.0,"vowel":0.0,"fricative":0.0,"plosive":0.0,"nasal":0.0}
 flux=np.array([float(f.get("flux",0)) for f in frames]);thr=float(np.median(flux)+3*max(float(np.median(np.abs(flux-np.median(flux)))),_EPS)) if n else 0.0
 c={"voiced":0,"vowel":0,"fricative":0,"plosive":0,"nasal":0};prev_burst=False
 for f in frames:
  voiced=bool(f.get("voiced")) or float(f.get("f0",f.get("pitch",0)))>0;silent=bool(f.get("silent"))
  b=float(f.get("bass",0));m=float(f.get("mid",0));h=float(f.get("high",0));tot=b+m+h+_EPS;flat=float(f.get("flatness",0));cent=float(f.get("centroid",f.get("brightness",0)))
  if voiced and not silent: c["voiced"]+=1
  if voiced and flat<0.3 and cent<2000: c["vowel"]+=1
  if h/tot>0.5 and flat>0.3: c["fricative"]+=1
  burst=float(f.get("flux",0))>=thr and float(f.get("energy",0))>0
  if burst and not prev_burst: c["plosive"]+=1
  prev_burst=burst
  if voiced and b/tot>0.6 and cent<500: c["nasal"]+=1
 return {k:float(v/n) for k,v in c.items()}
manner_fractions=manner_features
def _smooth(x,k=9):
 if len(x)<k: return x
 return np.convolve(x,np.ones(k)/k,mode="same")
def track_formants(samples,sample_rate=44100,fft_size=2048,hop=1024):
 x=np.asarray(samples,dtype=np.float64)
 if x.ndim>1: x=x[:,0]
 if x.size and np.max(np.abs(x))>1.5: x=x/32768.0
 sr=int(sample_rate);freqs=np.fft.rfftfreq(fft_size,1/sr);band=(freqs>=200)&(freqs<=3500);win=np.hanning(fft_size);out=[]
 if x.size==0: return out
 n=1+max(0,(len(x)-fft_size)//hop) if len(x)>=fft_size else 1;pad=np.zeros(n*hop+fft_size);pad[:len(x)]=x
 for i in range(n):
  blk=pad[i*hop:i*hop+fft_size]*win;mag=np.abs(np.fft.rfft(blk));log=np.log(mag+_EPS);sm=_smooth(log);cand=[]
  idx=np.where(band)[0]
  for j in idx:
   if 0<j<len(sm)-1 and sm[j]>=sm[j-1] and sm[j]>=sm[j+1]: cand.append((float(sm[j]),float(freqs[j])))
  cand.sort(reverse=True);picked=[]
  for val,fr in cand:
   if all(abs(fr-p[1])>=250 for p in picked): picked.append((val,fr))
   if len(picked)==3: break
  picked.sort(key=lambda p:p[1]);med=float(np.median(sm[band])) if np.any(band) else 0.0
  while len(picked)<3: picked.append((med,0.0))
  def conf(v): return float(min(1.0,max(0.0,(v-med)/(abs(v)+_EPS)))) if v>med else 0.0
  out.append({"time":float(i*hop/sr),"f1":float(picked[0][1]),"f2":float(picked[1][1]),"f3":float(picked[2][1]),"f1_confidence":conf(picked[0][0]),"f2_confidence":conf(picked[1][0]),"f3_confidence":conf(picked[2][0])})
 return out
def formant_transitions(track):
 out=[]
 for a,b in zip(track,track[1:]): out.append({"time":float(b["time"]),"df1":float(b["f1"]-a["f1"]),"df2":float(b["f2"]-a["f2"]),"df3":float(b["f3"]-a["f3"])})
 return out
def voice_onset_time(frames,sample_rate=44100):
 if not frames: return {"voice_onset_time_ms":None,"voicing_class":None,"burst_time":None,"voicing_time":None}
 flux=np.array([float(f.get("flux",0)) for f in frames]);thr=float(np.median(flux)+3*max(float(np.median(np.abs(flux-np.median(flux)))),_EPS))
 burst=None
 for i,f in enumerate(frames):
  if float(f.get("flux",0))>=thr and float(f.get("energy",0))>0: burst=i;break
 if burst is None: return {"voice_onset_time_ms":None,"voicing_class":None,"burst_time":None,"voicing_time":None}
 voicing=None
 for i in range(burst,len(frames)):
  if frames[i].get("voiced") or float(frames[i].get("f0",frames[i].get("pitch",0)))>0: voicing=i;break
 bt=float(frames[burst].get("time",burst*1024/sample_rate));vt=float(frames[voicing].get("time",voicing*1024/sample_rate)) if voicing is not None else None
 ms=None if vt is None else float((vt-bt)*1000.0)
 return {"voice_onset_time_ms":ms,"voicing_class":("voiced" if ms is not None and ms<30 else ("voiceless" if ms is not None else None)),"burst_time":bt,"voicing_time":vt}
def phonetic_summary(manner,mean_formants=None,vot=None):
 top=sorted(manner.items(),key=lambda kv:kv[1],reverse=True)[:3];txt=", ".join("%s %.2f"%(k,v) for k,v in top) if top else "no manner"
 s="Manner fractions: %s."%txt
 if mean_formants: s+=" Mean formants F1 %.0f Hz, F2 %.0f Hz, F3 %.0f Hz."%(mean_formants.get("f1",0),mean_formants.get("f2",0),mean_formants.get("f3",0))
 if vot and vot.get("voice_onset_time_ms") is not None: s+=" Voice onset time %.1f ms (%s). "%(vot["voice_onset_time_ms"],vot.get("voicing_class"))
 return s.strip()
def word_reading_sentence(match,manner=None,vot=None):
 cohort=(match or {}).get("cohort") or (match or {}).get("candidates") or [];nearest=(match or {}).get("name")
 parts=[]
 if nearest: parts.append("Nearest remembered word: %s (distance %.4f)."%(nearest,float((match or {}).get("distance") or 0)))
 if cohort: parts.append("Cohort: %s."%", ".join("%s %.4f"%(c["name"],float(c["distance"])) for c in cohort[:5]))
 if manner: parts.append(phonetic_summary(manner,None,vot))
 return " ".join(parts) if parts else "No remembered word matched."
def analyze_phonetics(frames,samples=None,sample_rate=44100,lexicon_match=None):
 manner=manner_features(frames);track=track_formants(samples,sample_rate) if samples is not None else []
 mean={"f1":float(np.mean([t["f1"] for t in track])) if track else 0.0,"f2":float(np.mean([t["f2"] for t in track])) if track else 0.0,"f3":float(np.mean([t["f3"] for t in track])) if track else 0.0}
 vot=voice_onset_time(frames,sample_rate);summary=phonetic_summary(manner,mean,vot)
 return {"manner":manner,"manner_fractions":manner,"top_cues":sorted(manner.items(),key=lambda kv:kv[1],reverse=True)[:3],"mean_formants":mean,"formant_track":track,"formant_transitions":formant_transitions(track),"voice_onset_time":vot,"voice_onset_time_ms":vot.get("voice_onset_time_ms"),"voicing_class":vot.get("voicing_class"),"summary":summary,"phonetic_summary":summary,"word_reading":word_reading_sentence(lexicon_match,manner,vot)}
