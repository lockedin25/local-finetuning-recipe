
import os
import json
from pathlib import Path
import mlx.core as mx
p=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'models/B_think/mlx/model.safetensors')
w=mx.load(str(p)); bad=[]; rows=[]
for k,v in w.items():
 x=v.astype(mx.float32); n=int(mx.sum(~mx.isfinite(x)).item())
 if n: bad.append([k,n])
 rows.append((k,list(v.shape),float(mx.sqrt(mx.mean(x*x)).item()),float(mx.max(mx.abs(x)).item())))
print(json.dumps({'tensors':len(w),'parameters':sum(v.size for v in w.values()),'nonfinite':bad,'largest_rms':sorted(rows,key=lambda r:r[2],reverse=True)[:12]}))
