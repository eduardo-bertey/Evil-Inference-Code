import torch


def fake_quantize_int4(w, group_size=32):
    in_feat, out_feat = w.shape
    gs = min(group_size, in_feat)
    pad = (gs - in_feat % gs) % gs
    if pad > 0:
        w_padded = torch.nn.functional.pad(w, (0, 0, 0, pad))
    else:
        w_padded = w
    num_groups = w_padded.shape[0] // gs
    w_grouped = w_padded.view(num_groups, gs, out_feat)
    scale = w_grouped.abs().amax(dim=1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-8)
    w_q = (w_grouped / scale).round().clamp(-8, 7) * scale
    w_q = w_q.view(-1, out_feat)[:in_feat]
    return w + (w_q - w).detach()


def fake_quantize_int8(w, group_size=32):
    in_feat, out_feat = w.shape
    gs = min(group_size, in_feat)
    pad = (gs - in_feat % gs) % gs
    if pad > 0:
        w_padded = torch.nn.functional.pad(w, (0, 0, 0, pad))
    else:
        w_padded = w
    num_groups = w_padded.shape[0] // gs
    w_grouped = w_padded.view(num_groups, gs, out_feat)
    scale = w_grouped.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    w_q = (w_grouped / scale).round().clamp(-128, 127) * scale
    w_q = w_q.view(-1, out_feat)[:in_feat]
    return w + (w_q - w).detach()


def quantize_params(model, precision="int4"):
    qfn = fake_quantize_int8 if precision == "int8" else fake_quantize_int4
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim == 2:
            param.data = qfn(param.data)
    return model
