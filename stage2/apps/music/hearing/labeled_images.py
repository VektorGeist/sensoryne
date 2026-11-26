"""stage2/apps/music/hearing/labeled_images.py - labeled pass: titles, axes, scales on pack images."""
import numpy as np
from PIL import Image, ImageDraw
__all__=["label_chladni_image","label_spectrogram_image","label_waveform_image","labeled_images_for_segment","relabel_pack"]
def _img(data,w=256,h=160):
 if isinstance(data,Image.Image): return data.convert("RGB").resize((w,h))
 a=np.asarray(data,dtype=np.float64)
 if a.size==0: a=np.zeros((h,w))
 if float(np.max(a))<=1.0: a=a*255
 return Image.fromarray(np.clip(a,0,255).astype(np.uint8),mode="L").convert("RGB").resize((w,h))
def _title(img,text):
 d=ImageDraw.Draw(img);d.rectangle([0,0,img.width,22],fill=(255,255,255));d.text((6,6),str(text)[:80],fill=(0,0,0));return img
def label_chladni_image(image,strongest_mode=None,pitch_hz=None,mode=None):
 mode=mode if mode is not None else strongest_mode;img=_title(_img(image),"Chladni mode %s pitch %s Hz"%(mode,pitch_hz));return img
def label_spectrogram_image(image,pitch_hz=None):
 img=_img(image);d=ImageDraw.Draw(img);d.line([30,10,30,img.height-20],fill=(0,0,0));d.line([30,img.height-20,img.width-5,img.height-20],fill=(0,0,0));d.text((2,10),"freq",fill=(0,0,0));d.text((img.width-70,img.height-16),"time",fill=(0,0,0));d.text((img.width-90,24),"dB",fill=(0,0,0));return _title(img,"Spectrogram pitch %s Hz"%pitch_hz)
def label_waveform_image(image,peak=None,rms=None,sample_rate=None):
 img=_img(image);d=ImageDraw.Draw(img);d.line([30,10,30,img.height-20],fill=(0,0,0));d.line([30,img.height-20,img.width-5,img.height-20],fill=(0,0,0));d.text((2,10),"amp",fill=(0,0,0));d.text((img.width-70,img.height-16),"time",fill=(0,0,0));return _title(img,"Waveform peak %s RMS %s"%(peak,rms))
def labeled_images_for_segment(segment):
 modes=segment.get("chladni_modes") or [];strong=modes[0].get("mode") if modes else None;pitch=segment.get("pitch_hz") or segment.get("harmonic_pitch");osc=segment.get("oscilloscope_block") or segment.get("oscilloscope") or {}
 return {"chladni":label_chladni_image(segment.get("chladni_image"),strong,pitch),"spectrogram":label_spectrogram_image(segment.get("spectrogram_image"),pitch),"waveform":label_waveform_image(osc.get("waveform_image"),osc.get("peak_amplitude"),osc.get("rms"))}
def relabel_pack(pack):
 import copy;out=copy.deepcopy(pack)
 for seg in out.get("segments_data",[]):
  labeled=labeled_images_for_segment(seg);seg["labeled_images"]={k:np.asarray(v).tolist() for k,v in labeled.items()};seg["chladni_image"]=seg["labeled_images"]["chladni"];seg["spectrogram_image"]=seg["labeled_images"]["spectrogram"]
 return out
