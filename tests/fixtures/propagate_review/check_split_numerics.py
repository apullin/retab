"""The channel split of probe_channel_split.py in eager torch: the original
graph's output shapes against those of rc4's (and the port's) Down result."""
import torch
y = torch.randn(1, 3, 5, 8)
ref = (y.permute(0, 3, 1, 2).clamp(0, 6)[:, 0:4])
a_ref, n_ref = ref.permute(0, 2, 3, 1), -ref
s = y.clamp(0, 6)[:, :, 0:4]
p = s.permute(0, 3, 1, 2)
a_rc4, n_rc4 = p.permute(0, 2, 3, 1), -p
print("reference outputs", tuple(a_ref.shape), tuple(n_ref.shape), "| rc4 Down outputs", tuple(a_rc4.shape), tuple(n_rc4.shape))
