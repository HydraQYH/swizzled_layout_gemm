# Swizzled Hierarchical Layout for GEMM

Convert the Layout of the operand matrix into a Swizzled Hierarchical Layout, so that each matrix block can be copied using the `cp.async.bulk` instruction instead of `cp.async.bulk.tensor`. We are very interested in the latency of these two approaches.
