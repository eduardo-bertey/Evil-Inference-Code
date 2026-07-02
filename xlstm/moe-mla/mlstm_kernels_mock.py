"""Mock for mlstm_kernels (native PyTorch fallback)."""
import sys, torch, torch.nn as nn
from dataclasses import dataclass

@dataclass
class mLSTMBackendConfig:
    chunkwise_kernel: str = "parallel"
    sequence_kernel: str = "native"
    step_kernel: str = "native"
    mode: str = "train"
    chunk_size: int = 64
    return_last_states: bool = True
    autocast_kernel_dtype: str = "float32"
    eps: float = 1e-6
    inference_state_dtype: str = "float32"

class mLSTMBackend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        from xlstm.blocks.mlstm import backends as native_backends
        self.native = native_backends

    def forward(self, q, k, v, i, f, c_initial=None, n_initial=None, m_initial=None):
        if i.ndim == 3: i = i.unsqueeze(-1)
        if f.ndim == 3: f = f.unsqueeze(-1)
        B, NH, S, DH = q.shape
        if S > 1 and c_initial is None:
            h = self.native.parallel_stabilized_simple(
                queries=q, keys=k, values=v, igate_preact=i, fgate_preact=f, eps=self.config.eps
            )
            # Also compute final state for MoE context
            dh_qk = q.shape[3]
            dh_v = v.shape[3]
            c = torch.zeros((B, NH, dh_qk, dh_v), device=q.device, dtype=q.dtype)
            n = torch.zeros((B, NH, dh_qk, 1), device=q.device, dtype=q.dtype)
            m = torch.zeros((B, NH, 1, 1), device=q.device, dtype=q.dtype)
            for t in range(S):
                _, (c, n, m) = self.native.recurrent_step_stabilized_simple(
                    c_state=c, n_state=n, m_state=m,
                    q=q[:, :, t:t+1], k=k[:, :, t:t+1], v=v[:, :, t:t+1],
                    igate_preact=i[:, :, t:t+1], fgate_preact=f[:, :, t:t+1], eps=self.config.eps
                )
            return h, (c, n, m)
        else:
            if c_initial is None:
                dh_qk, dh_v = q.shape[3], v.shape[3]
                c_initial = torch.zeros((B, NH, dh_qk, dh_v), device=q.device, dtype=q.dtype)
                n_initial = torch.zeros((B, NH, dh_qk, 1), device=q.device, dtype=q.dtype)
                m_initial = torch.zeros((B, NH, 1, 1), device=q.device, dtype=q.dtype)
            curr_c, curr_n, curr_m = c_initial, n_initial, m_initial
            h_list = []
            for t in range(S):
                h_t, (curr_c, curr_n, curr_m) = self.native.recurrent_step_stabilized_simple(
                    c_state=curr_c, n_state=curr_n, m_state=curr_m,
                    q=q[:, :, t:t+1], k=k[:, :, t:t+1], v=v[:, :, t:t+1],
                    igate_preact=i[:, :, t:t+1], fgate_preact=f[:, :, t:t+1], eps=self.config.eps
                )
                h_list.append(h_t)
            return torch.cat(h_list, dim=2), (curr_c, curr_n, curr_m)

def install_mock():
    mock_mod = type(sys)('mlstm_kernels')
    mock_mod_torch = type(sys)('mlstm_kernels.torch')
    mock_mod_backend = type(sys)('mlstm_kernels.torch.backend_module')
    sys.modules['mlstm_kernels'] = mock_mod
    sys.modules['mlstm_kernels.torch'] = mock_mod_torch
    sys.modules['mlstm_kernels.torch.backend_module'] = mock_mod_backend
    mock_mod_backend.mLSTMBackendConfig = mLSTMBackendConfig
    mock_mod_backend.mLSTMBackend = mLSTMBackend
    mock_mod_backend.ChunkwiseKernelType = str
    mock_mod_backend.SequenceKernelType = str
    mock_mod_backend.StepKernelType = str
    mock_mod_backend.DtypeType = str
    mock_mod_backend.BackendModeType = str
