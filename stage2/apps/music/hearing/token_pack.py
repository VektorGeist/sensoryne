"""stage2/apps/music/hearing/token_pack.py - 20 segment pack, oscilloscope, phonetics block, images.

The stored pack (build_token_pack) keeps everything, including
the pixel arrays, for the stored record and the viewer.  The
model-facing form (model_token_pack) carries every segment whole
with every reading, one name per reading, and no pixel arrays.
"""
import math, numpy as np
from PIL import Image
from .cochlea import Cochlea, load_wav
from .reading import aggregate_frames, read_frames
from .chladni import mode_weights, top_modes, draw_plate
from .field import place_time_field, render_field_image
from .harmonic import harmonic_pitch
from .phonetics import analyze_phonetics
__all__=["oscilloscope_block","build_token_pack","token_pack","make_token_pack","model_token_pack","model_facing_pack"]
_EPS=1e-12
def _waveform_array(samples,rows=32,cols=64):
 x=np.asarray(samples,dtype=np.float64)
 if x.ndim>1: x=x[:,0]
 out=np.zeros((rows,cols))
 if x.size==0: return out
 if np.max(np.abs(x))>1.5: x=x/32768.0
 edges=np.linspace(0,len(x),cols+1).astype(int)
 for c in range(cols):
  seg=x[edges[c]:edges[c+1]]
  if seg.size==0: continue
  lo=int(np.clip((1-float(np.min(seg)))/2*(rows-1),0,rows-1));hi=int(np.clip((1-float(np.max(seg)))/2*(rows-1),0,rows-1));out[min(lo,hi):max(lo,hi)+1,c]=1.0
 return out
def oscilloscope_block(samples,sample_rate=44100):
 x=np.asarray(samples,dtype=np.float64)
 if x.ndim>1: x=x[:,0]
 if x.size and np.max(np.abs(x))>1.5: x=x/32768.0
 sr=float(sample_rate);peak=float(np.max(np.abs(x))) if x.size else 0.0;rms=float(np.sqrt(np.mean(x**2))) if x.size else 0.0
 zcr=float(np.count_nonzero(np.diff(np.signbit(x))))/ (len(x)/sr) if len(x)>1 and sr>0 else 0.0
 env=[]
 if x.size:
  edges=np.linspace(0,len(x),21).astype(int)
  for a,b in zip(edges[:-1],edges[1:]): env.append(float(np.max(np.abs(x[a:b]))) if b>a else 0.0)
 else: env=[0.0]*20
 trend="steady"
 if len(env)>=4:
  first=float(np.mean(env[:len(env)//3]));last=float(np.mean(env[-len(env)//3:]))
  if last>first*1.2: trend="rising"
  elif last<first*0.8: trend="falling"
 return {"peak":peak,"rms":rms,"zero_crossing_rate":zcr,"peak_envelope_20bins":env,"trend":trend,"waveform_image":_waveform_array(x).tolist()}
def _model_oscilloscope(osc):
 if not isinstance(osc, dict):
  return osc
 return {"peak": osc.get("peak", osc.get("peak_amplitude")), "rms": osc.get("rms"), "zero_crossing_rate": osc.get("zero_crossing_rate", osc.get("zero_crossing_rate_per_sec")), "peak_envelope_20bins": list(osc.get("peak_envelope_20bins") or osc.get("envelope") or []), "trend": osc.get("trend")}
def _model_segment(seg):
 if not isinstance(seg, dict):
  return seg
 reading = seg.get("reading") if isinstance(seg.get("reading"), dict) else {}
 harm_candidates = seg.get("polyphony_candidates")
 if harm_candidates is None and isinstance(reading, dict):
  harm_candidates = []
 # Reading sentence: prefer segment summary/heard, else reading sentence.
 sentence = seg.get("summary") or seg.get("heard") or reading.get("sentence")
 # Harmonic pitch fields: segment carries harmonic_pitch (Hz), harmonic_midi, harmonic_confidence.
 harm_pitch = seg.get("harmonic_pitch")
 harm_midi = seg.get("harmonic_midi")
 harm_conf = seg.get("harmonic_confidence")
 # No harmonic pitch was heard when the segment carries none;
 # the Cochlea reading pitch is a different figure and must
 # not stand in for it.
 # Chladni modes: top modes list plus full weights as numbers.
 modes = seg.get("chladni_modes")
 weights = seg.get("mode_weights") or seg.get("chladni_weights")
 out = {
  "segment": seg.get("segment"),
  "start_time": seg.get("start_time"),
  "end_time": seg.get("end_time"),
  "reading": {
   "sentence": sentence,
   "note": reading.get("note", seg.get("note")),
   "cents": reading.get("cents", seg.get("cents")),
   "pitch_hz": reading.get("pitch_hz", seg.get("pitch_hz")),
   "loudness": reading.get("loudness", seg.get("loudness")),
   "brightness": reading.get("brightness", seg.get("brightness")),
   "energy_center_hz": reading.get("energy_center_hz"),
   "region": reading.get("region", seg.get("region")),
   "pitch_trend": reading.get("pitch_trend", seg.get("pitch_trend")),
   "loud_trend": reading.get("loud_trend", seg.get("loud_trend")),
   "onset_rate": reading.get("onset_rate", seg.get("onset_rate")),
   "strongest_modes": reading.get("strongest_modes"),
   "silent": reading.get("silent"),
   "tonal": reading.get("tonal"),
   "pitched": reading.get("pitched"),
  },
  "peak_band": seg.get("peak_band"),
  "chladni_modes": modes,
  "mode_weights": list(weights) if isinstance(weights, (list, tuple)) else weights,
  "harmonic_pitch_hz": harm_pitch,
  "harmonic_midi": harm_midi,
  "harmonic_confidence": harm_conf,
  "polyphony_candidates": harm_candidates,
  "gallery_match": seg.get("gallery_match"),
  "oscilloscope": _model_oscilloscope(seg.get("oscilloscope") or seg.get("oscilloscope_block") or {}),
 }
 return out
def model_token_pack(pack):
 """Return the model-facing form of a stored token pack.

 Every segment whole with every reading, one name per reading,
 no pixel arrays.  The stored pack is not mutated.
 """
 if not isinstance(pack, dict):
  return pack
 segments = pack.get("segments_data") or pack.get("segments") or []
 if isinstance(segments, int):
  segments = []
 model_segments = [_model_segment(s) for s in segments]
 phon = pack.get("phonetics") if isinstance(pack.get("phonetics"), dict) else {}
 # Normalise phonetics block: one name per figure.
 if isinstance(phon, dict):
  manner = phon.get("manner") or phon.get("manner_fractions") or {}
  model_phon = {
   "manner_fractions": manner,
   "top_cues": phon.get("top_cues"),
   "mean_formants": phon.get("mean_formants"),
   "voice_onset_time": phon.get("voice_onset_time"),
   "voice_onset_time_ms": phon.get("voice_onset_time_ms"),
  }
  # Any other phonetic figure the hearing produced stays under
  # its own name, except the duplicated manner alias.
  for k, v in phon.items():
   if k == "manner":
    continue
   model_phon.setdefault(k, v)
 else:
  model_phon = phon
 descriptor = pack.get("descriptor")
 weights = pack.get("mode_weights")
 out = {
  "segment_count": pack.get("segment_count", len(model_segments)),
  "segments_data": model_segments,
  "duration": pack.get("duration"),
  "sample_rate": pack.get("sample_rate"),
  "phonetics": model_phon,
  "descriptor": list(descriptor) if isinstance(descriptor, (list, tuple)) else descriptor,
  "mode_weights": list(weights) if isinstance(weights, (list, tuple)) else weights,
 }
 # Preserve any other pack-level descriptor figures without duplicating keys.
 for k, v in pack.items():
  if k in ("segments_data", "segments", "segment_count", "duration", "sample_rate", "phonetics", "descriptor", "mode_weights"):
   continue
  out.setdefault(k, v)
 return out
model_facing_pack = model_token_pack
def build_token_pack(path,num_segments=20,gallery=None,label_images=False,sample_rate=None):
 samples,sr=load_wav(path) if isinstance(path,(str,bytes)) else (np.asarray(path,dtype=np.float64),int(sample_rate or 44100))
 sr=int(sr);frames=Cochlea(sr).analyze(samples);agg=aggregate_frames(frames,duration_seconds=len(samples)/sr if sr else None);descriptor=agg["descriptor"];weights=mode_weights(descriptor)
 nseg=int(num_segments)
 if nseg<=0: raise ValueError("num_segments must be positive")
 bounds=np.linspace(0,len(frames),nseg+1).astype(int) if frames else [0]*(nseg+1);segments=[]
 for idx in range(nseg):
  a,b=int(bounds[idx]),int(bounds[idx+1]);seg_frames=frames[a:b] if frames else []
  if seg_frames: st=float(seg_frames[0]["time"]);et=float(seg_frames[-1]["time"])
  else: st=et=float(idx*(len(samples)/sr)/nseg) if sr and len(samples) else 0.0
  sa=int(np.clip(st*sr,0,len(samples)));sb=int(np.clip(et*sr,sa,len(samples)));seg_samples=samples[sa:sb]
  seg_agg=aggregate_frames(seg_frames) if seg_frames else agg;seg_desc=seg_agg.get("descriptor",descriptor);seg_weights=mode_weights(seg_desc);seg_reading=read_frames(seg_frames,mode_weights=seg_weights) if seg_frames else read_frames([],mode_weights=seg_weights)
  harm=harmonic_pitch(seg_samples,sr,cochlea_f0=seg_reading.get("f0") or seg_agg.get("median_pitch") or None)
  osc=oscilloscope_block(seg_samples,sr)
  match={"name":None,"nearest":None,"distance":None,"known":False,"candidates":[]}
  if gallery is not None and hasattr(gallery,"recognize"):
   from .trajectory import make_trajectory
   match=gallery.recognize(seg_desc,make_trajectory(seg_frames) if seg_frames else None)
  chladni_img=np.asarray(draw_plate(seg_weights,size=32)).tolist();spec_img=np.asarray(render_field_image(place_time_field(seg_frames))).tolist() if seg_frames else np.zeros((32,1)).tolist()
  segments.append({"segment":idx+1,"start_time":float(st),"end_time":float(et),"times":[float(st),float(et)],"reading":seg_reading,"summary":seg_reading.get("sentence"),"heard":seg_reading.get("sentence"),"pitch_hz":float(seg_reading.get("pitch_hz") or 0),"f0":float(seg_reading.get("f0") or 0),"note":seg_reading.get("note"),"cents":seg_reading.get("cents"),"loud":float(seg_reading.get("loudness") or seg_agg.get("loudness") or 0),"loudness":float(seg_reading.get("loudness") or seg_agg.get("loudness") or 0),"brightness":float(seg_reading.get("brightness") or seg_agg.get("brightness") or 0),"centroid":float(seg_reading.get("brightness") or seg_agg.get("brightness") or 0),"peak_band":seg_agg.get("peak_band"),"region":seg_reading.get("region"),"pitch_trend":seg_reading.get("pitch_trend"),"loud_trend":seg_reading.get("loud_trend"),"onset_flux":float(seg_agg.get("onset_rate") or 0),"onset_rate":float(seg_agg.get("onset_rate") or 0),"chladni_modes":top_modes(seg_weights,5),"mode_weights":seg_weights,"chladni_weights":seg_weights,"gallery_match":match,"harmonic_pitch":harm["harmonic_pitch_hz"],"harmonic_midi":harm["harmonic_midi"],"harmonic_confidence":harm["confidence"],"polyphony_candidates":harm["candidates"],"oscilloscope":osc,"oscilloscope_block":osc,"chladni_image":chladni_img,"spectrogram_image":spec_img,"descriptor":seg_desc})
 phon=analyze_phonetics(frames,samples,sr);pack={"segment_count":nseg,"segments":nseg,"segments_data":segments,"duration":float(len(samples)/sr) if sr else 0.0,"sample_rate":sr,"phonetics":{"manner_fractions":phon["manner"],"manner":phon["manner"],"top_cues":phon["top_cues"],"mean_formants":phon["mean_formants"],"voice_onset_time":phon["voice_onset_time"],"voice_onset_time_ms":phon["voice_onset_time_ms"]},"descriptor":descriptor,"mode_weights":weights}
 if label_images:
  from .labeled_images import relabel_pack;pack=relabel_pack(pack)
 return pack
token_pack=build_token_pack;make_token_pack=build_token_pack
