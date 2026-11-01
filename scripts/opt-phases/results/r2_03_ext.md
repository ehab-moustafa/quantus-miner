| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|-----------|-----------|-------|---------|
| 4-lane mat4, limb factored-sum | 2.5687 | 3.5757 | 0.7184 | Reject |
| 4-lane mat4, u128 factored-sum | 2.5687 | 2.4699 | 1.0400 | Reject |
| width-12 external, 3× limb mat4 + circulant fold | 13.7935 | 14.7085 | 0.9378 | Reject |
| width-12 external, fused u128 mat4×3 + circulant fold | 13.7935 | 12.2283 | 1.1280 | Adopt |
| width-12 external, custom::external_linear_layer | 13.7935 | 11.9415 | 1.1551 | Adopt |

Asserts passed (all-zero, all-one, 50k random 4-lane states, 20k random width-12 states, known mat4 and external vectors).

Old is the qp-poseidon-core 3.1.0 linear layer, timed in-process. Ratio is median old/new over 5 interleaved trials. The u128 mat4 row is Reject: a repeat of the same binary measured robust ratios 1.0285 and 1.0339, so the edge is not clearly above 1.03. `custom::external_linear_layer` stays faster than the fused u128 layer (11.9415 ns vs 12.2283 ns).
