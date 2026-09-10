import sys, os, torch, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/ai/lab/lingbot-gguf-r9700/ComfyUI_Rebels_LingBotWorld')
sys.path.insert(0,'/ai/lab/lingbot-gguf-r9700')
import gguf_bench as GB
GB.install_stubs('/ai/models/lingbot-gguf','/ai/models/lingbot-world-v2-14b-assets','/ai/lab/lingbot-gguf-r9700/input')
import lbworld_nodes as LB
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
print("flash2/3:", FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE, flush=True)
t0=time.perf_counter()
(b,)=LB.LBWorldLoader().load('LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf','Wan2.1_VAE.pth',6,2,pin_gb=2)
print("load %.1fs"%(time.perf_counter()-t0), flush=True)
m=b['model']
nw=sum(1 for x in m.modules() if isinstance(x,LB.GGUFLinearWrapper))
qb=sum(x.qdata.numel() for x in m.modules() if isinstance(x,LB.GGUFLinearWrapper))
dp=sum(p.numel()*p.element_size() for p in m.parameters() if p.device.type!='meta')
print("wrapped Linear:",nw)
print("host quant bytes %.2f GiB"%(qb/2**30))
print("dense (non-streamed) params %.2f GiB"%(dp/2**30))
print("meta left:", sum(1 for _,p in m.named_parameters() if p.device.type=='meta'))
print("pinned %.2f GiB"%(LB._pinned['bytes']/2**30))
print("GPU alloc %.2f reserved %.2f"%(torch.cuda.memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30))
r={}
for line in open('/proc/self/status'):
    if line.startswith(('VmRSS','VmSize','VmPin')): print("  ",line.strip())
