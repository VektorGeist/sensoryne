"""stage2/apps/story/invariants.py - declared invariants, reliability, coverage."""
import re
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
def _obj(art):
 if isinstance(art,dict) and isinstance(art.get("objective"),dict): return art["objective"]
 return {}
def evaluate_invariants(artifact,render_record,evidence=None):
 obj=_obj(artifact);secs=artifact.get("sections",[]) if isinstance(art:=artifact,dict) else [];checked=[]
 if "section_count" in obj:
  checked.append(_c("section_count","Delivered section count matches the stated section count.",len(secs)==obj["section_count"],len(secs),[] if len(secs)==obj["section_count"] else [{"stated":obj["section_count"],"measured":len(secs)}]))
  checked.append(_c("declared_section_count_match","The delivered story matches the section count stated in its objective, with both counts.",len(secs)==obj["section_count"],{"section_count":len(secs),"declared_section_count":obj["section_count"]},[] if len(secs)==obj["section_count"] else [{"stated":obj["section_count"],"measured":len(secs)}]))
 events=list(obj.get("promised_events") or [])
 if events:
  found=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): found=list(evidence["measurements"].get("promised_events_found") or [])
  else:
   text=" ".join(str(s.get("text","")) for s in secs if isinstance(s,dict)).lower();found=[e for e in events if str(e).lower() in text]
  missing=[e for e in events if e not in found];checked.append(_c("promised_events_present","Each promised event string appears in the delivered text.",not missing,found,missing))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"text analyzer","limits":["Promised events are checked as case-insensitive substrings of the delivered text.","Sentence counts split on sentence punctuation.","Dialogue share counts words inside double quotation marks.","Comparison section share tolerance is 0.05: a delivered section whose share of the delivered text departs from its written share by more than 0.05 is a section_share_off finding."]}
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
 obj=_obj(artifact);secs=artifact.get("sections",[]) if isinstance(artifact,dict) else []
 comp=_comparison(artifact,render_record,evidence)
 out=[]
 if isinstance(comp,dict) and comp.get("promised_events_total") is not None and comp.get("sections_written") is not None:
  events_delivered=comp.get("promised_events_delivered")
  events_total=comp.get("promised_events_total")
  sections_in_order=comp.get("sections_delivered_in_written_order")
  if sections_in_order is None: sections_in_order=comp.get("sections_in_written_order")
  sections_written=comp.get("sections_written")
 else:
  found=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): found=list(evidence["measurements"].get("promised_events_found") or [])
  elif isinstance(render_record,dict) and isinstance(render_record.get("measurements"),dict): found=list(render_record["measurements"].get("promised_events_found") or [])
  events_total=len(obj.get("promised_events") or [])
  events_delivered=sum(1 for e in (obj.get("promised_events") or []) if e in found)
  sections_written=len(secs)
  order=[]
  written_titles=[str(s.get("title","")) for s in secs if isinstance(s,dict)]
  delivered_order=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): delivered_order=list(evidence["measurements"].get("section_order") or [])
  for title in delivered_order:
   if title in written_titles:
    pos=written_titles.index(title)+1
    if pos not in order: order.append(pos)
  ordered=sorted(order)
  sections_in_order=sum(1 for i,p in enumerate(order) if p==ordered[i])
 out.append({"item":"promised events delivered, %s of %s"%(events_delivered,events_total),"covered":bool(events_total) and events_delivered==events_total})
 out.append({"item":"sections delivered in written order, %s of %s"%(sections_in_order,sections_written),"covered":bool(sections_written) and sections_in_order==sections_written})
 return out
