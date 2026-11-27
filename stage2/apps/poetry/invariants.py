"""stage2/apps/poetry/invariants.py - declared invariants, reliability, coverage."""
import copy
from .instrument import syllable_estimate
def _c(i,d,held,val,viol=None): return {"id":i,"description":d,"held":bool(held),"value":val,"violations":list(viol or [])}
def _lines(art):
 out=[]
 for s in art.get("stanzas",[]) if isinstance(art,dict) else []:
  if isinstance(s,list):
   for line in s: out.append(line)
 return out
def _form(art):
 if isinstance(art,dict):
  if isinstance(art.get("form"),dict): return art["form"]
  if isinstance(art.get("objective"),dict): return art["objective"]
 return {}
def evaluate_invariants(artifact,render_record,evidence=None):
 form=_form(artifact);stanzas=artifact.get("stanzas",[]) if isinstance(artifact,dict) else [];lines=_lines(artifact);checked=[]
 if "line_count" in form: checked.append(_c("line_count","Delivered line count matches the stated line count.",len(lines)==form["line_count"],len(lines),[] if len(lines)==form["line_count"] else [{"stated":form["line_count"],"measured":len(lines)}]))
 if "stanzas" in form: checked.append(_c("stanza_count","Delivered stanza count matches the stated stanza count.",len(stanzas)==form["stanzas"],len(stanzas),[] if len(stanzas)==form["stanzas"] else [{"stated":form["stanzas"],"measured":len(stanzas)}]))
 if form:
  declared_stanzas=form.get("stanzas");declared_lines=form.get("line_count")
  stanza_match=declared_stanzas is None or len(stanzas)==declared_stanzas
  line_match=declared_lines is None or len(lines)==declared_lines
  if declared_stanzas is not None or declared_lines is not None:
   checked.append(_c("declared_form_match","The delivered poem matches the form it declares, in stanza and line counts.",stanza_match and line_match,{"stanza_count":len(stanzas),"declared_stanzas":declared_stanzas,"line_count":len(lines),"declared_line_count":declared_lines},[] if stanza_match and line_match else [{"stated_stanzas":declared_stanzas,"measured_stanzas":len(stanzas),"stated_lines":declared_lines,"measured_lines":len(lines)}]))
 target=form.get("syllables_per_line")
 if target is not None:
  estimates=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict) and isinstance(evidence["measurements"].get("lines"),list):
   estimates=[e.get("syllable_estimate") for e in evidence["measurements"]["lines"]]
  else: estimates=[sum(syllable_estimate(w) for w in str(line).split()) for line in lines]
  bad=[{"line":i+1,"estimate":v,"target":target} for i,v in enumerate(estimates) if v is not None and v!=target]
  checked.append(_c("syllable_estimates_match_target","Per-line syllable estimates match the stated syllables per line; counts are estimates.",not bad,estimates,bad))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"mechanical text analyzer","limits":["Syllable counts are estimates by the stated counting rule: count vowel groups, subtract a silent final e, floor of one per word.","Rhyme grouping uses spelled endings normalized from the last vowel.","The analyzer reads delivered text only."]}
def _comparison_of(evidence,render_record):
 for source in (evidence,render_record):
  if isinstance(source,dict):
   comp=source.get("comparison")
   if isinstance(comp,dict): return comp
   full=source.get("full")
   if isinstance(full,dict) and isinstance(full.get("comparison"),dict): return full["comparison"]
 return None
def objective_coverage(artifact,render_record,evidence=None):
 form=_form(artifact);stanzas=artifact.get("stanzas",[]) if isinstance(artifact,dict) else [];lines=_lines(artifact);out=[]
 comp=_comparison_of(evidence,render_record)
 if isinstance(comp,dict) and comp.get("lines_written") is not None and comp.get("stanzas_written") is not None:
  delivered_as_written=comp.get("lines_delivered_as_written")
  lines_written=comp.get("lines_written")
  stanzas_complete=comp.get("stanzas_complete")
  stanzas_written=comp.get("stanzas_written")
 else:
  delivered_as_written=len(lines)
  lines_written=len(lines)
  stanzas_complete=len(stanzas)
  stanzas_written=len(stanzas)
 out.append({"item":"lines delivered as written, %s of %s"%(delivered_as_written,lines_written),"covered":bool(lines_written) and delivered_as_written==lines_written})
 out.append({"item":"stanzas complete, %s of %s"%(stanzas_complete,stanzas_written),"covered":bool(stanzas_written) and stanzas_complete==stanzas_written})
 return out
