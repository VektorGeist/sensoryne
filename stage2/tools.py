"""stage2/tools.py - ten tool definitions (seven for the working model, three for the judge) and dispatch. No app logic."""
import copy
OPEN_SESSION="sensoryne_open_session";SUBMIT_WORK="sensoryne_submit_work";SUBMIT_DECISION="sensoryne_submit_decision";FINISH_SESSION="sensoryne_finish_session";PROBE="sensoryne_probe";RECALL="sensoryne_recall";JUDGE_LIST="sensoryne_judge_list_sessions";JUDGE_OPEN="sensoryne_judge_open";JUDGE_VERDICT="sensoryne_judge_verdict";RESUME="sensoryne_resume_session"
TOOL_NAMES=(OPEN_SESSION,SUBMIT_WORK,SUBMIT_DECISION,FINISH_SESSION,PROBE,RECALL,RESUME,JUDGE_LIST,JUDGE_OPEN,JUDGE_VERDICT)
_DECISION={"type":"object","additionalProperties":True,"required":["faults","revisions","complete","assessment"],"properties":{"faults":{"type":"array","items":{"type":"object","additionalProperties":True,"required":["description","where"],"properties":{"description":{"type":"string"},"where":{"type":"string"}}}},"revisions":{"type":"array","items":{"type":"object","additionalProperties":True}},"complete":{"type":"boolean"},"assessment":{"type":"string"}}}
TOOLS=[
 {"type":"function","name":OPEN_SESSION,"description":"Open a session, create its record and store entry, return session id, composition prompt text, and control information rules when control.","parameters":{"type":"object","additionalProperties":False,"required":["application","objective","condition","test"],"properties":{"application":{"type":"string"},"objective":{},"condition":{"type":"string","enum":["perceptual","control"]},"test":{"type":"boolean"}}}},
 {"type":"function","name":SUBMIT_WORK,"description":"Submit a work. The artifact is passed as a JSON object value: an object with instrument and parts, each part holding notes whose velocity is a number from 0 to 1. Validate, store version 1, render, perceive, return evidence/structure/input summary or control structure/artifact.","parameters":{"type":"object","additionalProperties":False,"required":["session_id","artifact"],"properties":{"session_id":{"type":"string"},"artifact":{"type":"object"}}}},
 {"type":"function","name":SUBMIT_DECISION,"description":"Submit a decision. The decision is passed as a JSON object value carrying faults, revisions, complete, assessment. Record it, apply revisions, store new version, re-render, re-perceive, return evidence, structure, applied/rejected, session so far.","parameters":{"type":"object","additionalProperties":False,"required":["session_id","decision"],"properties":{"session_id":{"type":"string"},"decision":_DECISION}}},
 {"type":"function","name":FINISH_SESSION,"description":"Declare the work complete. End as model_complete and build report.","parameters":{"type":"object","additionalProperties":False,"required":["session_id"],"properties":{"session_id":{"type":"string"}}}},
 {"type":"function","name":PROBE,"description":"Apply a decision's revisions to a throwaway copy, render and perceive it, return hypothetical evidence marked probe true with a delta against the current version. No version is created and the session cannot end.","parameters":{"type":"object","additionalProperties":False,"required":["session_id","decision"],"properties":{"session_id":{"type":"string"},"decision":_DECISION}}},
 {"type":"function","name":JUDGE_LIST,"description":"Judge seat: list sessions from disk. The judge reads a session's complete record and renders a judgment of the work itself; the judge cannot submit work, revise, or end a maker's session.","parameters":{"type":"object","additionalProperties":False,"required":[],"properties":{}}},
 {"type":"function","name":JUDGE_OPEN,"description":"Judge seat: open a finished session's complete record pack. The judge reads a session's complete record and renders a judgment of the work itself; the judge cannot submit work, revise, or end a maker's session.","parameters":{"type":"object","additionalProperties":False,"required":["session_id"],"properties":{"session_id":{"type":"string"}}}},
 {"type":"function","name":JUDGE_VERDICT,"description":"Judge seat: record a judgment of the work itself in a session's history. The judge reads a session's complete record and renders a judgment of the work itself; the judge cannot submit work, revise, or end a maker's session.","parameters":{"type":"object","additionalProperties":False,"required":["session_id","verdict"],"properties":{"session_id":{"type":"string"},"verdict":{"type":"object","required":["bottom_line","judgment","strengths","weaknesses"],"properties":{"bottom_line":{"type":"string","enum":["good","mixed","not_good"]},"judgment":{"type":"string"},"strengths":{"type":"array","items":{"type":"string"}},"weaknesses":{"type":"array","items":{"type":"string"}}}}}}},
 {"type":"function","name":RECALL,"description":"Return stored pairs for an application from sessions that ended model_complete, most recent first, up to limit.","parameters":{"type":"object","additionalProperties":False,"required":["application","limit"],"properties":{"application":{"type":"string"},"limit":{"type":"integer"}}}},
 {"type":"function","name":RESUME,"description":"Resume a session from its record on disk, rebuilding its live state under its existing id so work and decisions can continue on it. Refuses a session that ended as model_complete and errors on an unknown session id.","parameters":{"type":"object","additionalProperties":False,"required":["session_id"],"properties":{"session_id":{"type":"string"}}}},
]
class ToolRegistry:
 def __init__(self): self.sessions={}
 def register(self,sid,ctx): self.sessions[sid]=ctx;return ctx
 def get(self,sid):
  ctx=self.sessions.get(sid)
  if ctx is None: raise ValueError("unknown session_id: %r"%(sid,))
  return ctx
def dispatch(reg,name,args):
 from . import controller
 if name not in TOOL_NAMES: raise ValueError("unknown tool: %r"%(name,))
 args=copy.deepcopy(args or {})
 if name==OPEN_SESSION: return controller.open_session(reg,args)
 if name==RECALL: return controller.recall_memory(reg,args.get("application"),args.get("limit"))
 if name==RESUME: return controller.resume_session(reg,args.get("session_id"))
 from . import judge as judge_module
 if name==JUDGE_LIST: return judge_module.list_sessions(reg)
 if name==JUDGE_OPEN: return judge_module.open_judgment(reg,args.get("session_id"))
 if name==JUDGE_VERDICT: return judge_module.record_verdict(reg,args.get("session_id"),args.get("verdict"))
 sid=args.get("session_id");ctx=reg.get(sid)
 if ctx.get("ended_by") is not None: return {"session_id":sid,"ended_by":ctx["ended_by"],"message":"session has ended; no further tool calls accepted"}
 if name==SUBMIT_WORK: return controller.submit_work(ctx,args.get("artifact"))
 if name==SUBMIT_DECISION: return controller.submit_decision(ctx,args.get("decision"))
 if name==PROBE: return controller.probe(ctx,args.get("decision"))
 return controller.finish_session(ctx)
