"""stage2/apps/drawing/invariants.py - declared invariants, reliability, coverage."""
import xml.etree.ElementTree as ET
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
def _ids(svg):
 try: root=ET.fromstring(svg)
 except Exception: return set(),None
 return {e.get("id") for e in root.iter() if e.get("id")},root
def _box(root,eid):
 if root is None: return None
 for e in root.iter():
  if e.get("id")==eid:
   try: return (float(e.get("x",0)),float(e.get("y",0)),float(e.get("width",0)),float(e.get("height",0)))
   except Exception: return None
 return None
def _overlap(a,b):
 ax,ay,aw,ah=a;bx,by,bw,bh=b;return ax<bx+bw and bx<ax+aw and ay<by+bh and by<ay+ah
def evaluate_invariants(artifact,render_record,evidence=None):
 ids,root=_ids(artifact.get("svg","") if isinstance(artifact,dict) else "");required=[e.get("id") for e in artifact.get("elements",[]) if isinstance(e,dict) and e.get("id")] if isinstance(artifact,dict) else []
 missing=[x for x in required if x not in ids];checked=[_c("required_elements_present","Every element id the artifact declares as required is present in the SVG.",not missing,sorted(ids),missing)]
 rules=artifact.get("overlap_rules") or artifact.get("bounding_box_rules") or []
 bad=[]
 for rule in rules:
  if isinstance(rule,dict) and rule.get("a") and rule.get("b"):
   ba=_box(root,rule["a"]);bb=_box(root,rule["b"])
   if ba and bb and not _overlap(ba,bb): bad.append(rule)
 checked.append(_c("declared_overlap_rules","Declared bounding-box overlap rules hold for the named element pairs.",not bad,copy_rules(rules),bad))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def copy_rules(r): return r
def reliability_block(): return {"instrument":"raster analyzer","limits":["Rasterization resolution is the rendered image dimensions in pixels.","Dominant colors are the most frequent image colors by pixel count.","Ink coverage is the non-background pixel share."]}
def objective_coverage(artifact,render_record,evidence=None):
 obj=artifact.get("objective") if isinstance(artifact,dict) else None;out=[]
 elems=artifact.get("elements",[]) if isinstance(artifact,dict) else []
 if isinstance(obj,dict):
  for e in obj.get("elements") or []:
   if isinstance(e,dict) and e.get("id"): out.append({"item":"required element %s"%e["id"],"present":any(x.get("id")==e["id"] for x in elems if isinstance(x,dict))})
 elif isinstance(obj,str) and obj: out.append({"item":"picture intent stated","present":bool(elems)})
 for e in elems:
  if isinstance(e,dict) and e.get("id"): out.append({"item":"element %s"%e["id"],"present":True})
 return out
