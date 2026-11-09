import os
for k,v in {"HF_HOME":"./.cache/huggingface","TORCH_HOME":"./.cache/torch","XDG_CACHE_HOME":"./.cache","MPLCONFIGDIR":"./.cache/mpl","WANDB_DISABLED":"true"}.items():
    os.environ.setdefault(k, os.path.abspath(v) if v.startswith(".") else v)
import torch, transformers, trl, peft
from transformers import AutoTokenizer, AutoModelForCausalLM
S,T="Qwen/Qwen3.5-0.8B","Qwen/Qwen3.5-9B"
print(torch.__version__,transformers.__version__,trl.__version__,peft.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
ts,tt=AutoTokenizer.from_pretrained(S),AutoTokenizer.from_pretrained(T)
print("vocab equal:",ts.get_vocab()==tt.get_vocab(),"len",len(ts),len(tt),"special equal:",ts.all_special_tokens==tt.all_special_tokens)
for n,t in((S,ts),(T,tt)):
    m=AutoModelForCausalLM.from_pretrained(n,dtype=torch.bfloat16).cuda().eval()
    print(n,"logit dim",m.config.get_text_config().vocab_size,"params",sum(p.numel() for p in m.parameters())/1e9)
    msgs=[{"role":"user","content":"Hi, when will my deposit be refunded?"}]
    p=t.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    print(repr(p))
    ids=t(p,return_tensors="pt").to("cuda")
    out=m.generate(**ids,max_new_tokens=60,do_sample=False)
    print("REPLY:",t.decode(out[0,ids.input_ids.shape[1]:],skip_special_tokens=True))
print("peak GB",torch.cuda.max_memory_allocated()/1e9)
