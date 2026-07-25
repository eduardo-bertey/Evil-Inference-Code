import torch, time, gc, sys
sys.path.insert(0, r"C:\Users\Emabe\Documents\GitHub\xlstm\rust\python - sparse")
from sparse import SparseAttentionMio3
from attention import Attention

BSZ = 128
S, K = 4096, 2
torch.manual_seed(0)
x = torch.randn(1, S, 768)

HD = 64
NH, NK = 12, 4
NB = max(1, (S + BSZ - 1) // BSZ)
est_full_mem = NH * S * S * 4 // (1024**2)
est_sparse_mem = NK * NB * BSZ * HD * 4 // (1024**2)
est_topk_gather = NK * S * K * BSZ * HD * 4 // (1024**2)
print(f"S={S}  BSZ={BSZ}  NB={NB}  topK={K}  ratio={K*100//NB}%")
print(f"  full_mat ~{est_full_mem}MB  topk_gather ~{est_topk_gather}MB  kv_cache ~{est_sparse_mem}MB")

gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
t0 = time.perf_counter()
with torch.no_grad(): a = Attention(768, NH, NK, HD).eval()(x)
mem_gqa = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
print(f"GQA:   {1000*(time.perf_counter()-t0):.0f}ms  {mem_gqa:.0f}MB  sum={a.sum():.2f}")

gc.collect(); torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
t0 = time.perf_counter()
with torch.no_grad(): b = SparseAttentionMio3(768, NH, NK, HD, num_selected_blocks=K).eval()(x)
mem_mio = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
print(f"Mio3K={K}: {1000*(time.perf_counter()-t0):.0f}ms  {mem_mio:.0f}MB  sum={b.sum():.2f}")
print(f"diff_max={((a-b).abs().max().item()):.4f}")
