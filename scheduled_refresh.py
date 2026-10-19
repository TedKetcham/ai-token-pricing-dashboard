"""Scheduled research; never silently replaces approved pricing with unreviewed LLM output."""
import json,os
from pathlib import Path
import pandas as pd
from research import collect_documents,propose_updates,DATA
root=Path(__file__).resolve().parent
docs=collect_documents()
(root/'refresh_status.json').write_text(json.dumps([{'provider':d['provider'],'url':d['url'],'fetched_at_utc':d['fetched_at_utc'],'status':d['status']} for d in docs],indent=2))
key=os.getenv('OPENAI_API_KEY')
if key:
    proposals,errors=propose_updates(key,docs,pd.read_csv(DATA))
else:
    proposals=[];errors=['OPENAI_API_KEY GitHub Actions secret not set; documents fetched but price extraction skipped.']
(root/'pending_price_review.json').write_text(json.dumps({'proposals':proposals,'errors':errors,'human_approval_required':True},indent=2))
print('Sources fetched:',sum(d['status']=='ok' for d in docs),'/',len(docs),'candidate rows:',len(proposals))
