"""stage2/apps/argument/invariants.py - declared invariants, reliability, coverage."""
import re
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
def _obj(art):
 if isinstance(art,dict) and isinstance(art.get("objective"),dict): return art["objective"]
 return {}
def evaluate_invariants(artifact,render_record,evidence=None):
 obj=_obj(artifact);claims=artifact.get("claims",[]) if isinstance(art:=artifact,dict) else [];checked=[]
 if "claim_count" in obj:
  checked.append(_c("claim_count","Delivered claim count matches the stated claim count.",len(claims)==obj["claim_count"],len(claims),[] if len(claims)==obj["claim_count"] else [{"stated":obj["claim_count"],"measured":len(claims)}]))
  checked.append(_c("declared_claim_count_match","The delivered argument matches the claim count stated in its objective, with both counts.",len(claims)==obj["claim_count"],{"claim_count":len(claims),"declared_claim_count":obj["claim_count"]},[] if len(claims)==obj["claim_count"] else [{"stated":obj["claim_count"],"measured":len(claims)}]))
 unsupported=[c.get("id") for c in claims if isinstance(c,dict) and not c.get("premises") and not c.get("sources")]
 checked.append(_c("claims_supported","Every claim carries at least one premise or one cited source; no claim is asserted with nothing under it.",not unsupported,unsupported,unsupported))
 concl=artifact.get("conclusion") if isinstance(artifact,dict) else None
 checked.append(_c("conclusion_named","The conclusion names an existing claim.",concl in [c.get("id") for c in claims if isinstance(c,dict)],concl,[] if concl in [c.get("id") for c in claims if isinstance(c,dict)] else [{"conclusion":concl}]))
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"argument analyzer","limits":["Presence is by key phrase: a claim counts as delivered only where its exact key phrase appears in the delivered text.","Figure comparison is over stated numbers, dates, and percentages: only figures the claim and the delivered statement or cited passage state are compared.","Definition conflicts are found where the delivered text states a definition in the form 'term is/means/refers to ...'.","The circularity judgment compares a conclusion's text with its premises' texts, by normalized equality or word-set overlap; the stated threshold is 0.9, named in the reliability block."]}
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
 obj=_obj(artifact);claims=artifact.get("claims",[]) if isinstance(artifact,dict) else []
 comp=_comparison(artifact,render_record,evidence)
 out=[]
 if isinstance(comp,dict) and comp.get("claims_total") is not None and comp.get("fact_claims_total") is not None:
  claims_delivered=comp.get("claims_delivered")
  claims_total=comp.get("claims_total")
  fact_checked=comp.get("fact_claims_checked")
  fact_total=comp.get("fact_claims_total")
  conclusion_delivered=comp.get("conclusion_delivered")
 else:
  found=[]
  if isinstance(evidence,dict) and isinstance(evidence.get("measurements"),dict): found=list(evidence["measurements"].get("claims_present") or [])
  elif isinstance(render_record,dict) and isinstance(render_record.get("measurements"),dict): found=list(render_record["measurements"].get("claims_present") or [])
  claims_total=len(claims)
  claims_delivered=sum(1 for c in claims if isinstance(c,dict) and c.get("id") in found)
  fact_total=sum(1 for c in claims if isinstance(c,dict) and c.get("kind")=="fact")
  checks=[]
  if isinstance(comp,dict) and isinstance(comp.get("source_checks"),list): checks=comp["source_checks"]
  fact_checked=sum(1 for ch in checks if isinstance(ch,dict) and ch.get("status")!="unverified")
  conclusion_delivered=artifact.get("conclusion") in found if isinstance(artifact,dict) else False
 out.append({"item":"claims delivered, %s of %s"%(claims_delivered,claims_total),"covered":bool(claims_total) and claims_delivered==claims_total})
 out.append({"item":"fact claims checked, %s of %s"%(fact_checked,fact_total),"covered":bool(fact_total) and fact_checked==fact_total})
 out.append({"item":"conclusion delivered, %s"%("yes" if conclusion_delivered else "no"),"covered":bool(conclusion_delivered)})
 return out
