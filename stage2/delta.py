"""stage2/delta.py - pure measurement comparison between two versions."""
import copy, math
def _num(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v))
def _j(v):
 if isinstance(v,dict): return {str(k):_j(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [_j(x) for x in v]
 if hasattr(v,"tolist"): return _j(v.tolist())
 if isinstance(v,(bool,str,int,float)) or v is None: return v
 return str(v)
def _walk(a,b,path,changed,added,removed):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b),key=str):
   p=f"{path}.{k}" if path else str(k)
   if k not in a: added.append(p)
   elif k not in b: removed.append(p)
   else: _walk(a[k],b[k],p,changed,added,removed)
 elif isinstance(a,list) and isinstance(b,list):
  for i in range(max(len(a),len(b))):
   p=f"{path}.{i}" if path else str(i)
   if i>=len(a): added.append(p)
   elif i>=len(b): removed.append(p)
   else: _walk(a[i],b[i],p,changed,added,removed)
 else:
  if a!=b or type(a)!=type(b):
   e={"path":path,"before":_j(a),"after":_j(b)}
   if _num(a) and _num(b):
    e["absolute_delta"]=float(b)-float(a)
    if float(a)!=0: e["relative_delta"]=(float(b)-float(a))/float(a)
   changed.append(e)
def _sdiff(a,b,path,added,removed,modified):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b),key=str):
   p=f"{path}.{k}" if path else str(k)
   if k not in a: added.append(p)
   elif k not in b: removed.append(p)
   else: _sdiff(a[k],b[k],p,added,removed,modified)
 elif isinstance(a,list) and isinstance(b,list):
  for i in range(max(len(a),len(b))):
   p=f"{path}.{i}" if path else str(i)
   if i>=len(a): added.append(p)
   elif i>=len(b): removed.append(p)
   else: _sdiff(a[i],b[i],p,added,removed,modified)
 else:
  if a!=b or type(a)!=type(b): modified.append(path)
def compute_delta(from_version,to_version,before_m,after_m,before_s,after_s):
 bm=_j(before_m or {});am=_j(after_m or {})
 if not isinstance(bm,dict): bm={}
 if not isinstance(am,dict): am={}
 changed,added,removed=[],[],[]
 _walk(bm,am,"",changed,added,removed)
 unchanged=sorted([k for k in bm if k in am and bm[k]==am[k]],key=str)
 sa,sr,sm=[],[],[];_sdiff(before_s or {},after_s or {},"",sa,sr,sm)
 return {"from_version":from_version,"to_version":to_version,
  "changed_measurements":sorted(changed,key=lambda e:e["path"]),
  "added_measurements":sorted(added),"removed_measurements":sorted(removed),
  "unchanged_keys":unchanged,
  "structural_diff":{"added":sorted(sa),"removed":sorted(sr),"modified":sorted(sm)}}
measurements_delta=compute_delta
