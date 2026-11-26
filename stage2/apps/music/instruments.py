"""stage2/apps/music/instruments.py - instrument table. Ranges, written+sounding, variants, note table B0(11)-C8(108) A4=440."""
import math
__all__=["NOTE_TABLE","NOTES","INSTRUMENTS","midi_of_name","note_frequency","note_ref","nearest_note","get_instrument","instrument_names"]
_PC=[("C","C"),("C#","Db"),("D","D"),("D#","Eb"),("E","E"),("F","Fb"),("F#","Gb"),("G","G"),("G#","Ab"),("A","A"),("A#","Bb"),("B","Cb")]
_CI={"C":0,"B#":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"Fb":4,"F":5,"E#":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,"Cb":11}
def _parse(name):
 if not isinstance(name,str) or len(name)<2: raise ValueError("invalid note name: %r"%(name,))
 l=name[0].upper();r=name[1:]
 if r.startswith("#"): a,o=1,r[1:]
 elif r.startswith("b"): a,o=-1,r[1:]
 else: a,o=0,r
 if l not in _CI or not o.lstrip("-").isdigit(): raise ValueError("invalid note name: %r"%(name,))
 return (int(o)+1)*12+_CI[l]+a
def midi_of_name(n): return _parse(n)
def note_frequency(n): return 440.0*(2.0**((midi_of_name(n)-69)/12.0))
def note_ref(n):
 n=str(n);return {"name":n,"midi":midi_of_name(n),"freq_hz":note_frequency(n)}
def _build():
 t={}
 for midi in range(11,109):
  o=midi//12-1;f=440.0*(2.0**((midi-69)/12.0))
  for c in _PC[midi%12]:
   fn="%s%d"%(c,o);t[fn]={"name":fn,"midi":midi,"frequency_hz":f,"freq_hz":f}
 return t
NOTE_TABLE=_build()
NOTES=sorted({e["midi"]:{"name":e["name"],"midi":e["midi"],"frequency_hz":e["frequency_hz"],"freq_hz":e["freq_hz"]} for e in NOTE_TABLE.values()}.values(),key=lambda e:e["midi"])
def nearest_note(f):
 if not f or f<=0: raise ValueError("frequency must be positive")
 e=69.0+12.0*math.log2(float(f)/440.0);m=int(round(e))
 return {"name":"%s%d"%(_PC[m%12][0],m//12-1),"midi":m,"frequency_hz":440.0*(2.0**((m-69)/12.0)),"measured_hz":float(f),"cents":(e-m)*100.0}
def _range(lo,hi): return {"low":note_ref(lo),"high":note_ref(hi),"low_midi":midi_of_name(lo),"high_midi":midi_of_name(hi)}
def _entry(n,t,written=None,sounding=None,**k):
 e={"name":n,"transposition":t,"written":written,"sounding":sounding};e.update(k);return e
def _oct_up(name,octaves):
 m=midi_of_name(name)+12*octaves;return "%s%d"%(_PC[m%12][0],m//12-1)
_GT={"standard":["E2","A2","D3","G3","B3","E4"],"drop_d":["D2","A2","D3","G3","B3","E4"],"open_g":["D2","G2","D3","G3","B3","D4"],"open_d":["D2","A2","D3","F#3","A3","D4"],"dadgad":["D2","A2","D3","G3","A3","D4"]}
_GTOP={24:"E6",22:"D6",20:"B5",19:"Bb5"}
def _guitar():
 tun={}
 for tn,os_ in _GT.items():
  lo=os_[0];hi_str=os_[-1];s_hi=_oct_up(hi_str,2);w_lo=_oct_up(lo,1);w_hi=_oct_up(s_hi,1)
  tun[tn]={"open_strings":[note_ref(n) for n in os_],"sounding":_range(lo,s_hi),"written":_range(w_lo,w_hi)}
 return _entry("guitar","written one octave above sounding",written=_range("E3","E7"),sounding=_range("E2","E6"),tuning="standard",tunings=tun,open_strings=[note_ref(n) for n in _GT["standard"]],fret_count=24,top_note_by_fret={str(k):note_ref(v) for k,v in _GTOP.items()},variants={n:{"tunings":tun[n]} for n in _GT if n!="standard"})
INSTRUMENTS={
 "piano":_entry("piano","non-transposing",sounding=_range("A0","C8"),absolute_range=_range("A0","C8"),key_count=88,first_key=note_ref("A0"),last_key=note_ref("C8")),
 "guitar":_guitar(),
 "violin":_entry("violin","non-transposing",sounding=_range("G3","E7"),full_top=note_ref("E7"),practical_top=note_ref("A6"),open_strings=[note_ref(n) for n in ["G3","D4","A4","E5"]]),
 "viola":_entry("viola","non-transposing",sounding=_range("C3","E6"),full_top=note_ref("E6"),practical_top=note_ref("A5"),open_strings=[note_ref(n) for n in ["C3","G3","D4","A4"]]),
 "cello":_entry("cello","non-transposing",sounding=_range("C2","A5"),full_top=note_ref("A5"),practical_top=note_ref("E5"),open_strings=[note_ref(n) for n in ["C2","G2","D3","A3"]]),
 "double bass":_entry("double bass","sounds one octave below written",written=_range("E2","D5"),sounding=_range("E1","D4"),practical_top={"sounding":note_ref("G3"),"written":note_ref("G4")},open_strings=[note_ref(n) for n in ["E1","A1","D2","G2"]],variants={"low_c_extension":{"sounding_low":note_ref("C1")},"five_string":{"sounding_low":note_ref("B0")}}),
 "harp":_entry("harp","non-transposing",sounding=_range("Cb1","Gb7"),absolute_range=_range("Cb1","Gb7"),string_count=47,pedal_position="natural",open_strings_scale="C flat major: Cb Db Eb Fb Gb Ab Bb repeating each octave",pedals=["D","C","B","E","F","G","A"],pedal_positions={"up":"flat","middle":"natural","down":"sharp"},no_pedal_strings=[{"name":"Cb1","note":note_ref("Cb1"),"tuning":"natural, by hand"},{"name":"Db1","note":note_ref("Db1"),"tuning":"natural, by hand"},{"name":"Gb7","note":note_ref("Gb7"),"tuning":"natural, by hand"}]),
 "flute":_entry("flute","non-transposing",sounding=_range("C4","D7"),variants={"b_footjoint":{"low":note_ref("B3")}}),
 "oboe":_entry("oboe","non-transposing",sounding=_range("Bb3","A6")),
 "clarinet":_entry("clarinet","sounds a major second below written",written=_range("E3","C7"),sounding=_range("D3","Bb6"),key="Bb",variants={"low_eb_key":{"written_low":note_ref("Eb3"),"sounding_low":note_ref("Db3")}}),
 "bassoon":_entry("bassoon","non-transposing",sounding=_range("Bb1","E5"),variants={"bell_extension":{"low":note_ref("A1"),"removes":note_ref("Bb1")}}),
 "trumpet":_entry("trumpet","sounds a major second below written",written=_range("F#3","C6"),sounding=_range("E3","Bb5"),key="Bb"),
 "french horn":_entry("french horn","sounds a perfect fifth below written",written=_range("F#2","C6"),sounding=_range("B1","F5"),key="F",variants={"low_f_variant":{"written_low":note_ref("F2"),"sounding_low":note_ref("Bb1")}}),
 "tenor trombone":_entry("tenor trombone","non-transposing",sounding=_range("E2","Bb4"),pedal_tones="excluded",variants={"f_attachment":{"low":note_ref("C2")}}),
 "soprano saxophone":_entry("soprano saxophone","sounds a major second below written",written=_range("Bb3","F6"),sounding=_range("Ab3","Eb6"),key="Bb",high_f_sharp_key={"written":note_ref("F#6")}),
 "alto saxophone":_entry("alto saxophone","sounds a major sixth below written",written=_range("Bb3","F6"),sounding=_range("Db3","Ab5"),key="Eb",high_f_sharp_key={"written":note_ref("F#6")}),
 "tenor saxophone":_entry("tenor saxophone","sounds a major ninth below written",written=_range("Bb3","F6"),sounding=_range("Ab2","Eb5"),key="Bb",high_f_sharp_key={"written":note_ref("F#6")}),
 "baritone saxophone":_entry("baritone saxophone","sounds an octave and a major sixth below written",written=_range("Bb3","F6"),sounding=_range("Db2","Ab4"),key="Eb",high_f_sharp_key={"written":note_ref("F#6")},variants={"low_a_key":{"written_low":note_ref("A3"),"sounding_low":note_ref("C2")}}),
}
def get_instrument(n):
 if not isinstance(n,str) or n not in INSTRUMENTS: raise ValueError("unknown instrument: %r"%(n,))
 return INSTRUMENTS[n]
def instrument_names(): return sorted(INSTRUMENTS)
