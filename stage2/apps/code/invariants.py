"""stage2/apps/code/invariants.py - declared invariants, reliability, coverage."""
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
def evaluate_invariants(artifact,render_record,evidence=None):
 files=artifact.get("files",[]) if isinstance(artifact,dict) else [];tests=artifact.get("tests",[]) if isinstance(artifact,dict) else [];purpose=artifact.get("purpose") if isinstance(artifact,dict) else None
 original=(render_record or {}).get("original_artifact") if isinstance(render_record,dict) else None
 original_purpose=original.get("purpose") if isinstance(original,dict) else purpose
 defined=[t for t in tests if isinstance(t,dict) and "expected" in t and t.get("expected") is not None]
 checked=[_c("tests_still_defined","Every stated test is still defined with an expected output.",len(defined)==len(tests) and bool(tests),len(tests),[] if defined else [{"stated":len(tests),"defined":0}]),
 _c("purpose_unchanged","The purpose string is unchanged from the submitted artifact.",purpose==original_purpose and bool(purpose),purpose,[] if purpose==original_purpose else [{"current":purpose,"original":original_purpose}]),
 _c("entry_point_present","The entry-point file is present among the delivered files.",bool(files),files[0].get("name") if files and isinstance(files[0],dict) else None,[] if files else [{"files":0}])]
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"execution","limits":["Timeout is 10 seconds per stated test.","Output matches expected only on exact character-for-character equality.","Execution runs in a temporary directory with reduced environment and disabled sockets; it is not a security sandbox and is not for untrusted external use."]}
def objective_coverage(artifact,render_record,evidence=None):
 out=[];purpose=artifact.get("purpose") if isinstance(artifact,dict) else None;tests=artifact.get("tests",[]) if isinstance(artifact,dict) else []
 if purpose: out.append({"item":"purpose stated","present":True})
 for t in tests:
  if isinstance(t,dict): out.append({"item":"stated test %s"%(t.get("name") or ""),"present":"expected" in t and t.get("expected") is not None})
 return out
