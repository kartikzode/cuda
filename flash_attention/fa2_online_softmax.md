# Flash Attention-2: Understanding the Online Softmax Decomposition

## The Core Problem: Why We Can't Just Tile Softmax

In standard attention, the softmax is a **global operation**:

```
For row i: softmax(S[i, :]) = exp(S[i, :] - max(S[i, :])) / sum(exp(S[i, :] - max(S[i, :])))
```

This requires **knowing all values in the row** to compute:
1. The row-wise maximum (for numerical stability)
2. The sum of exponentials (for normalization)

If you tile the computation (process keys/values in blocks), you can't compute the softmax for each block independently—you need to see the entire row of scores across **all** key/value blocks.

The breakthrough insight is: **You can incrementally compute softmax across blocks using online statistics (m, ℓ).**

---

## Online Softmax Decomposition (The Mathematical Trick)

### Standard Softmax (Full View)

Imagine you have attention scores split into two blocks:

```
S = [S^(1) | S^(2)]    where S^(1), S^(2) ∈ R^(Br × Bc)
V = [V^(1) | V^(2)]    where V^(1), V^(2) ∈ R^(Bc × d)
```

To compute the output, standard softmax would do:

```
Step 1: m = max(rowmax(S^(1)), rowmax(S^(2)))      # Global max across all keys
Step 2: ℓ = rowsum(exp(S^(1) - m)) + rowsum(exp(S^(2) - m))  # Global sum across all keys
Step 3: P = diag(ℓ)^(-1) [exp(S^(1) - m); exp(S^(2) - m)]   # Global softmax matrix
Step 4: O = P @ [V^(1); V^(2)]                     # Final output
```

This requires **knowing both S^(1) and S^(2) at once** to compute m and ℓ.

### Online Softmax (Incremental View)

Flash Attention-2 instead processes blocks **sequentially** and updates statistics:

```
=== Process First Block ===
m^(1) = rowmax(S^(1))                           # Max of first block
ℓ^(1) = rowsum(exp(S^(1) - m^(1)))             # Sum of first block (unnormalized)
P̃^(1) = diag(ℓ^(1))^(-1) exp(S^(1) - m^(1))   # Unnormalized softmax for first block
O^(1) = P̃^(1) @ V^(1)                          # Partially accumulated output

=== Process Second Block ===
m^(2) = max(m^(1), rowmax(S^(2)))  # *** KEY: Update running max ***
```

The crucial update for the maximum:
```
m^(2) = max(m^(1), rowmax(S^(2)))
```

Now you have the **true global maximum** after seeing the second block.

### The Rescaling Trick

Once you have the new max m^(2), you need to rescale the previous output because the softmax denominator changed:

**Old denominator** (with local max m^(1)): ℓ^(1) = Σ exp(S^(1) - m^(1))
**New denominator** (with global max m^(2)): ℓ^(2) = Σ exp(S^(1) - m^(2)) + Σ exp(S^(2) - m^(2))

Using the exponential rescaling property:

```
ℓ^(2) = exp(m^(1) - m^(2)) · ℓ^(1) + rowsum(exp(S^(2) - m^(2)))
```

This is the **online softmax update equation**. You're adjusting the first block's contribution based on the fact that you found a larger max value.

### Updating the Output

The output is updated as:

```
O̭^(2) = diag(exp(m^(1) - m^(2)))^(-1) O^(1) + exp(S^(2) - m^(2)) V^(2)
         ↑                                       ↑
    Rescales old output              Adds new block's contribution

O^(2) = diag(ℓ^(2))^(-1) Õ^(2)    # Final normalization at the end
```

Breaking this down:
- First term: Rescale the accumulated output from the first block
- Second term: Add the weighted values from the second block
- Final division: Normalize by the true softmax denominator

---

## Why This Works: A Concrete Example

Let's trace through with actual numbers:

### First Block
```
S^(1) = [[2.0, 3.0],
         [1.5, 2.5]]   (shape: 2×2)

V^(1) = [[1, 0],
         [0, 1]]       (shape: 2×2)

m^(1) = rowmax(S^(1)) = [3.0, 2.5]
P̃^(1) = exp(S^(1) - m^(1)) = [[exp(-1), exp(0)],    = [[0.368, 1.0],
                                [exp(-1), exp(0)]]       [0.368, 1.0]]
ℓ^(1) = rowsum(P̃^(1)) = [1.368, 1.368]

O^(1) = P̃^(1) @ V^(1) = [[0.368, 1.0],
                          [0.368, 1.0]]
```

### Second Block
```
S^(2) = [[4.0, 2.0],
         [2.0, 3.0]]   (shape: 2×2)

V^(2) = [[1, 0],
         [0, 1]]       (shape: 2×2)

m^(2) = max(m^(1), rowmax(S^(2))) = max([3.0, 2.5], [4.0, 3.0])
      = [4.0, 3.0]   # *** Key insight: max updated! ***

ℓ^(2) = exp(m^(1) - m^(2)) ⊙ ℓ^(1) + rowsum(exp(S^(2) - m^(2)))
      = exp([3.0, 2.5] - [4.0, 3.0]) ⊙ [1.368, 1.368]
        + rowsum(exp([[4.0, 2.0], [2.0, 3.0]] - [4.0, 3.0]))
      = exp([-1.0, -0.5]) ⊙ [1.368, 1.368]
        + rowsum([[1.0, exp(-2)], [exp(-1), 1.0]])
      = [0.368, 0.606] ⊙ [1.368, 1.368] + [1.135, 1.368]
      = [0.503, 0.829] + [1.135, 1.368]
      = [1.638, 2.197]

P̃^(2) = exp(S^(2) - m^(2)) = [[exp(0), exp(-2)],    = [[1.0, 0.135],
                                [exp(-1), exp(0)]]       [0.368, 1.0]]

O̭^(2) = diag(exp(m^(1) - m^(2)))^(-1) O^(1) + P̃^(2) V^(2)
       = diag(exp([-1.0, -0.5]))^(-1) [[0.368, 1.0], [0.368, 1.0]]
         + [[1.0, 0.135], [0.368, 1.0]] @ [[1, 0], [0, 1]]

       = diag([0.368, 0.606])^(-1) [[0.368, 1.0], [0.368, 1.0]]
         + [[1.0, 0.135], [0.368, 1.0]]

       = [[1.0/0.368, 1.0/0.368], [1.0/0.606, 1.0/0.606]] ⊙ [[0.368, 1.0], [0.368, 1.0]]
         + [[1.0, 0.135], [0.368, 1.0]]

       = [[1.0, 2.716], [0.606, 1.648]] + [[1.0, 0.135], [0.368, 1.0]]
       = [[2.0, 2.851], [0.974, 2.648]]

O^(2) = diag(ℓ^(2))^(-1) O̭^(2)
      = diag([1/1.638, 1/2.197]) [[2.0, 2.851], [0.974, 2.648]]
      = [[1.221, 1.741], [0.443, 1.205]]
```

This final O^(2) is **identical** to what you'd get if you computed the full softmax across both blocks from the start!

---

## Flash Attention-2: Algorithmic Optimization

The paper describes two key optimizations to the online softmax:

### Optimization 1: Don't Rescale Every Iteration

**Original (FA-1)**: After each block, divide by the softmax denominator:

```
O^(j) = diag(ℓ^(j))^(-1) O̭^(j)    # Rescale after each block
```

**FA-2 Optimization**: Keep an "unnormalized" version and only rescale at the end:

```
Õ^(j) = diag(exp(m^(j-1) - m^(j)))^(-1) Õ^(j-1) + exp(S^(j) - m^(j)) V^(j)
        # *** No normalization by ℓ^(j) ***

O = diag(ℓ^(final))^(-1) Õ^(final)   # Normalize only once at the very end
```

**Why?** Each non-matmul FLOP (like the division by ℓ) is **16× slower** than a matmul FLOP on modern GPUs. By deferring all rescaling to the very end, you reduce expensive scalar operations.

### Optimization 2: Use LogSumExp Instead of Separate m, ℓ

**Original (FA-1)**: Store both:
- `m[i]`: row-wise maximum
- `ℓ[i]`: row-wise sum of exponentials

**FA-2 Optimization**: Store only LogSumExp:

```
L[i] = m[i] + log(ℓ[i])   # Combined: log(Σ exp(S[i] - m[i]))
```

**Why?** 
1. Saves memory: One vector instead of two
2. Sufficient for backward pass: You can recover everything needed
3. Numerically more stable in log-space

---

## Flash Attention-2 Forward Pass Algorithm

From the paper (Algorithm 1):

```
for i in 1 to Tr:  # For each query block
    Load Q_i from HBM
    Initialize O_i^(0) = 0, ℓ_i^(0) = 0, m_i^(0) = -∞
    
    for j in 1 to Tc:  # For each key/value block
        Load K_j, V_j from HBM
        
        # Compute scores
        S_ij = Q_i @ K_j^T              # Shape: (Br × Bc)
        
        # Update running max (ONLINE SOFTMAX STEP 1)
        m_ij = rowmax(S_ij)
        m_i^(j) = max(m_i^(j-1), m_ij)
        
        # Compute exp(S - m)
        P̃_ij = exp(S_ij - m_ij)
        
        # Update running normalization factor (ONLINE SOFTMAX STEP 2)
        ℓ_ij = rowsum(P̃_ij)
        ℓ_i^(j) = exp(m_i^(j-1) - m_i^(j)) ⊙ ℓ_i^(j-1) + ℓ_ij
        
        # Update output with rescaling (ONLINE SOFTMAX STEP 3)
        O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j)))^(-1) O_i^(j-1) + P̃_ij @ V_j
    
    # Final normalization (happens once per query block)
    O_i = diag(ℓ_i^(Tc))^(-1) O_i^(Tc)
    L_i = m_i^(Tc) + log(ℓ_i^(Tc))      # Store logsumexp for backward
    
    Write O_i, L_i to HBM
```

---

## The Decomposition Property: Why It's Called "Associative"

The online softmax rescaling is **associative**, meaning:

```
softmax([block_1 | block_2 | block_3])
= combine(combine(softmax(block_1), softmax(block_2)), softmax(block_3))
```

This is NOT true for softmax itself (softmax is not associative), but it IS true for the **rescaled online softmax**. This associative property is exploited in later work (e.g., LeanAttention) to enable even more parallelism.

---

## Memory Savings Summary

### Standard Attention (FA-0)
- **Must materialize**: S (N×N), P (N×N)
- **Memory**: O(N²)
- **Problem**: Can't tile—need full matrices for softmax

### Flash Attention (FA-1)
- **Must materialize**: m (N), ℓ (N)
- **Memory**: O(N)
- **Approach**: Online softmax with two statistics

### Flash Attention-2 (FA-2)
- **Must materialize**: L = m + log(ℓ) (N)
- **Memory**: O(N)
- **Improvement**: Combined statistic, reduced non-matmul FLOPs

---

## Key Insights

1. **Softmax decomposition**: You can compute softmax incrementally across blocks if you maintain running max and sum statistics.

2. **Exponential property**: `exp(a) + exp(b) = exp(max(a,b)) [exp(a - max(a,b)) + exp(b - max(a,b))]`
   
   This is the mathematical foundation of the rescaling trick.

3. **No approximation**: The online softmax produces **exact** results—identical to standard softmax, just computed differently.

4. **GPU optimization**: By deferring normalization and using combined statistics, FA-2 reduces expensive non-matmul FLOPs.

5. **Associativity**: The rescaling operation is associative, enabling further parallelism opportunities in future variants.