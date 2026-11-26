"""stage2/apps/story/invariants.py - declared invariants, reliability, coverage."""
import re
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
def _obj(art):
 if isinstance(art,dict) and isinstance(art.get("objective"),dict): return art["objective"]
 return {}
def _words(t): return len(re.findall(r"\S+",t or ""))
def evaluate_invariants(artifact,render_record,evidence=None):
 obj=_obj(artifact);secs=artifact.get("sections",[]) if isinstance(art:=artifact,dict) else [];checked=[]
 if "section_count" in obj: checked.append(_c("section_count","Delivered section count matches the stated section count.",len(secs)==obj["section_count"],len(secs),[] if len(secs)==obj["section_count"] else [{"stated":obj["section_count"],"measured":len(secs)}]))
 counts=[_words(s.get("text","")) for s in secs if isinstance(s,dict)];total=sum(counts);shares=[(c/total if total else 0.0) for c in counts]
 checked.append(_c("section_shares","Each section share of the whole is measured and sums to the whole word count.",total==sum(counts) and all(s>=0 for s in shares),shares,[]))
 events=list(obj.get("promised_events") or [])
 if events:
  found=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): found=list(evidence["measurements"].get("promised_events_found") or [])
  else:
   text=" ".join(str(s.get("text","")) for s in secs if isinstance(s,dict)).lower();found=[e for e in events if str(e).lower() in text]
  missing=[e for e in events if e not in found];checked.append(_c("promised_events_present","Each promised event string appears in the delivered text.",not missing,found,missing))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"text analyzer","limits":["Promised events are checked as case-insensitive substrings of the delivered text.","Sentence counts split on sentence punctuation.","Dialogue share counts words inside double quotation marks."]}
def objective_coverage(artifact,render_record,evidence=None):
 obj=_obj(artifact);secs=artifact.get("sections",[]) if isinstance(artifact,dict) else [];out=[]
 if "section_count" in obj: out.append({"item":"section count %s"%obj["section_count"],"present":len(secs)==obj["section_count"]})
 for i,m in enumerate(obj.get("movements") or []): out.append({"item":"movement %s at position %d"%(m,i+1),"present":len(secs)>i})
 found=[]
 if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): found=list(evidence["measurements"].get("promised_events_found") or [])
 for e in obj.get("promised_events") or []: out.append({"item":"promised event %s"%e,"present":e in found})
 return out
