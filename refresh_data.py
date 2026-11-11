import json
from pathlib import Path
from app import dashboard
p=Path(__file__).parent/'data'/'dashboard.json'
p.parent.mkdir(exist_ok=True)
data=dashboard.__wrapped__() if hasattr(dashboard,'__wrapped__') else dashboard()
p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved daily market snapshot:',p)
