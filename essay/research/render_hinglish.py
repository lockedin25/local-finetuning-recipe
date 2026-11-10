"""Render unedited replies from the new, synthetic Hinglish probe."""
from pathlib import Path
import json,runpy
O=Path(__file__).resolve().parent
ns=runpy.run_path(str(O/'render_examples.py'))
ns['render'].__globals__['rows']=json.loads((O/'hinglish_examples.json').read_text())
ns['render']('A_nothink','hinglish',4,'Hinglish words, but the conversation is lost','example_hinglish.png')
print('Rendered Hinglish example.')
