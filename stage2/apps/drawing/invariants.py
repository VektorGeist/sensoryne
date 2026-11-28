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
 viewbox=root.get("viewBox") if root is not None else None
 checked.append(_c("viewbox_present","Whether a viewBox is present on the SVG root is measured and recorded.",bool(viewbox),viewbox,[] if viewbox else [{"viewBox":viewbox}]))
 present={x:(x in ids) for x in required}
 rules=artifact.get("overlap_rules") or artifact.get("bounding_box_rules") or []
 bad=[]
 for rule in rules:
  if isinstance(rule,dict) and rule.get("a") and rule.get("b"):
   ba=_box(root,rule["a"]);bb=_box(root,rule["b"])
   # A rule naming an absent element cannot hold: the pair is not
   # present to overlap, so the judgment is not held.
   if ba is None or bb is None:
    bad.append(rule)
   elif not _overlap(ba,bb): bad.append(rule)
 checked.append(_c("declared_overlap_rules","Declared bounding-box overlap rules hold for the named element pairs.",not bad,rules,bad))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"raster analyzer","limits":["Rasterization resolution is the rendered image dimensions in pixels.","Dominant colors are the most frequent image colors by pixel count.","Ink coverage is the non-background pixel share.","Comparison geometry tolerance is 0.5 SVG units: a delivered shape whose position or size departs from its declared values by more than 0.5 is a geometry_off finding."]}
def _comparison_of(evidence,render_record):
 for source in (evidence,render_record):
  if isinstance(source,dict):
   comp=source.get("comparison")
   if isinstance(comp,dict): return comp
   full=source.get("full")
   if isinstance(full,dict) and isinstance(full.get("comparison"),dict): return full["comparison"]
 return None
def _comparison(artifact,render_record,evidence):
 comp=_comparison_of(evidence,render_record)
 if isinstance(comp,dict): return comp
 try:
  from .comparison import compute_comparison
  return compute_comparison(artifact,render_record,evidence)
 except Exception:
  return None
def objective_coverage(artifact,render_record,evidence=None):
 comp=_comparison(artifact,render_record,evidence)
 if isinstance(comp,dict) and comp.get("shapes_realized_as_declared") is not None and comp.get("shapes_declared") is not None:
  realized=comp.get("shapes_realized_as_declared")
  declared=comp.get("shapes_declared")
 else:
  declared=len([e.get("id") for e in artifact.get("elements",[]) if isinstance(e,dict) and e.get("id")]) if isinstance(artifact,dict) else 0
  realized=0
 return [{"item":"declared shapes realized as declared, %s of %s"%(realized,declared),"covered":bool(declared) and realized==declared}]
