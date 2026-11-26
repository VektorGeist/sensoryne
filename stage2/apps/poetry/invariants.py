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
def objective_coverage(artifact,render_record,evidence=None):
 form=_form(artifact);lines=_lines(artifact);out=[]
 if "line_count" in form: out.append({"item":"line count %s"%form["line_count"],"present":len(lines)==form["line_count"]})
 if "stanzas" in form: out.append({"item":"stanza count %s"%form["stanzas"],"present":len(artifact.get("stanzas",[]))==form["stanzas"]})
 if "syllables_per_line" in form: out.append({"item":"syllables per line %s"%form["syllables_per_line"],"present":bool(lines)})
 for pair in form.get("rhyme_pairs",[]) if isinstance(form.get("rhyme_pairs"),list) else []: out.append({"item":"rhyme pair %s"%pair,"present":bool(pair) and max(pair)<=len(lines)})
 return out
