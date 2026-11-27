"""stage2/apps/music/hearing/test_sounds.py - deterministic synthetic exam sounds: tone, chord, sweep, noise, click."""
import numpy as np
__all__=["generate_tone","generate_chord","generate_sweep","generate_noise","generate_click","build_exam_set","make_exam_set"]
def _t(duration,sample_rate):
 n=int(round(float(duration)*int(sample_rate)));return np.arange(n),n
def generate_tone(freq,duration=1.0,sample_rate=44100,amplitude=0.5):
 t,n=_t(duration,sample_rate);return np.asarray(amplitude*np.sin(2*np.pi*float(freq)*t/float(sample_rate)),dtype=np.float64)
def generate_chord(freqs,duration=1.0,sample_rate=44100,amplitude=0.4):
 t,n=_t(duration,sample_rate);x=np.zeros(n)
 for f in freqs: x+=np.sin(2*np.pi*float(f)*t/float(sample_rate))
 return x*(float(amplitude)/max(1,len(list(freqs))))
def generate_sweep(f_start,f_end,duration=1.0,sample_rate=44100,amplitude=0.5):
 t,n=_t(duration,sample_rate);sr=float(sample_rate);f0=float(f_start);f1=float(f_end)
 if n==0: return np.zeros(0)
 freq=f0+(f1-f0)*(t/n);phase=2*np.pi*np.cumsum(freq)/sr;return amplitude*np.sin(phase)
def generate_noise(duration=1.0,sample_rate=44100,amplitude=0.3,seed=0):
 t,n=_t(duration,sample_rate);rng=np.random.default_rng(int(seed));return amplitude*rng.standard_normal(n)
def generate_click(duration=0.1,sample_rate=44100,amplitude=0.8,position=0.0):
 t,n=_t(duration,sample_rate);x=np.zeros(n);idx=int(np.clip(float(position)*float(sample_rate),0,max(0,n-1)))
 if n: x[idx]=float(amplitude)
 return x
def build_exam_set(sample_rate=44100):
 sr=int(sample_rate);return [{"name":"tone_220","kind":"tone","freq_hz":220.0,"sample_rate":sr,"samples":generate_tone(220,1.0,sr).tolist()},{"name":"tone_440","kind":"tone","freq_hz":440.0,"sample_rate":sr,"samples":generate_tone(440,1.0,sr).tolist()},{"name":"chord_c_major","kind":"chord","freq_hz":[261.63,329.63,392.0],"sample_rate":sr,"samples":generate_chord([261.63,329.63,392.0],1.0,sr).tolist()},{"name":"sweep_200_800","kind":"sweep","freq_hz":[200.0,800.0],"sample_rate":sr,"samples":generate_sweep(200,800,1.0,sr).tolist()},{"name":"noise","kind":"noise","sample_rate":sr,"samples":generate_noise(1.0,sr).tolist()},{"name":"click","kind":"click","sample_rate":sr,"samples":generate_click(0.1,sr).tolist()}]
make_exam_set=build_exam_set
