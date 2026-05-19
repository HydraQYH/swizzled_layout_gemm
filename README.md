# Swizzled Hierarchical Layout for GEMM

Convert the Layout of the operand matrix into a Swizzled Hierarchical Layout, so that each matrix block can be copied using the `cp.async.bulk` instruction instead of `cp.async.bulk.tensor`. We are very interested in the latency of these two approaches.

```
# b -- mn_major
python3 dense_gemm_persistent.py --mnkl 1,4096,4096,8 --cluster_shape_mn 1,2 --b_major n
# b -- k_major
python3 dense_gemm_persistent.py --mnkl 1,4096,4096,8 --cluster_shape_mn 1,2 --b_major k
```
