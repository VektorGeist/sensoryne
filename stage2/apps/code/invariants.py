"""stage2/apps/code/invariants.py - declared invariants, reliability, coverage."""
def _c(i,d,h,v,viol=None): return {"id":i,"description":d,"held":bool(h),"value":v,"violations":list(viol or [])}
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
def evaluate_invariants(artifact,render_record,evidence=None):
 files=artifact.get("files",[]) if isinstance(artifact,dict) else [];tests=artifact.get("tests",[]) if isinstance(artifact,dict) else [];purpose=artifact.get("purpose") if isinstance(artifact,dict) else None
 original=(render_record or {}).get("original_artifact") if isinstance(render_record,dict) else None
 original_purpose=original.get("purpose") if isinstance(original,dict) else purpose
 defined=[t for t in tests if isinstance(t,dict) and "expected" in t and t.get("expected") is not None]
 stated_tests=len(tests) if isinstance(tests,list) else 0
 expected_count=len(defined)
 has_purpose=isinstance(purpose,str) and bool(purpose.strip())
 purpose_held=purpose==original_purpose and bool(purpose)
 tests_defined_held=len(defined)==len(tests) and bool(tests)
 # Entry point is checked against the delivered files when the render
 # record carries them, otherwise against the written files.
 delivered_files=None
 if isinstance(render_record,dict) and isinstance(render_record.get("files"),list):
  delivered_files=[f for f in render_record.get("files") if isinstance(f,str)]
 if delivered_files is not None:
  entry_files=delivered_files
  entry_value="main.py" if "main.py" in entry_files else (sorted(entry_files)[0] if entry_files else None)
 else:
  entry_files=[f.get("name") for f in files if isinstance(f,dict) and isinstance(f.get("name"),str)]
  entry_value=files[0].get("name") if files and isinstance(files[0],dict) else None
 entry_held=bool(entry_files)
 checked=[_c("tests_still_defined","Every stated test is still defined with an expected output.",tests_defined_held,len(tests),[] if tests_defined_held else [{"stated":len(tests),"defined":len(defined)}]),
 _c("test_count_stated","At least one test is stated.",stated_tests>0,stated_tests,[] if stated_tests>0 else [{"stated":0}]),
 _c("tests_with_expected_output","At least one stated test states an expected output.",expected_count>0,expected_count,[] if expected_count>0 else [{"expected":0,"stated":stated_tests}]),
 _c("purpose_stated","A purpose is stated.",has_purpose,bool(purpose) if isinstance(purpose,str) else purpose,[] if has_purpose else [{"purpose":purpose}]),
 _c("purpose_unchanged","The purpose string is unchanged from the submitted artifact.",purpose_held,purpose,[] if purpose_held else [{"current":purpose,"original":original_purpose}]),
 _c("entry_point_present","The entry-point file is present among the delivered files.",entry_held,entry_value,[] if entry_held else [{"files":len(entry_files)}])]
 return {"checked":checked,"summary":{"total_checked":len(checked),"held":sum(1 for c in checked if c["held"]),"broken":sum(1 for c in checked if not c["held"])}}
def reliability_block(): return {"instrument":"execution","limits":["Timeout is 10 seconds per stated test.","Output matches expected only on exact character-for-character equality.","Execution runs in a temporary directory with reduced environment and disabled sockets; it is not a security sandbox and is not for untrusted external use."]}
def objective_coverage(artifact,render_record,evidence=None):
 comp=_comparison(artifact,render_record,evidence)
 if isinstance(comp,dict) and comp.get("checks_passed") is not None and comp.get("checks_written") is not None:
  passed=comp.get("checks_passed");written=comp.get("checks_written")
 elif isinstance(comp,dict) and comp.get("tests_passed") is not None and comp.get("tests_total") is not None:
  passed=comp.get("tests_passed");written=comp.get("tests_total")
 else:
  tests=artifact.get("tests",[]) if isinstance(artifact,dict) else [];written=len(tests) if isinstance(tests,list) else 0;passed=0
 return [{"item":"checks passed, %s of %s"%(passed,written),"covered":bool(written) and passed==written}]
