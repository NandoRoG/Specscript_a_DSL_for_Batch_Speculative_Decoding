# Specscript_a_DSL_for_Batch_Speculative_Decoding

## Quick Start
The entire project runs in a single Google Colab notebook on a free T4 GPU. No local setup required.
**Runtime requirement:** Go to `Runtime → Change runtime type → T4 GPU` before running any cell.

## Understanding the Results
 
### Why throughput is below the AR baseline
 
Two independent reasons. Understanding both is important for anyone extending this work.
 
**Reason 1 — Model pair acceptance rate.**
`OPT-125M` and `OPT-1.3b` were not trained together as a draft-target pair. The token acceptance rate α is low, meaning the draft is wrong often and the speculation overhead nearly cancels the gain. The speedup formula from Leviathan et al. (ICML 2023) is:
 
```
speedup = (1 - α^(k+1)) / (1 - α)
```
 
When α ≈ 0, the numerator collapses regardless of implementation quality. This is a property of the model pair, not of SpecScript. Zhang et al. use Vicuna-7B with a 68M draft model specifically trained to predict the target's outputs — that pair achieves α ≈ 0.7 and shows real speedup.
 
**Reason 2 — Simplified KV-cache management.**
The full EQSPEC algorithm shifts existing KV-cache entries by δᵢ positions per sequence after each repad. This project resets the cache from scratch each round, which is correct but means verification cost grows with sequence length rather than staying constant. Full cache shifting is the primary future work item.

## Extending This Project
 
### Try a better model pair
 
The single highest-impact change is swapping in a properly matched draft-target pair. Replace these lines in Cell 4 and Cell 8:
 
```python
# Current (low acceptance rate)
DRAFT_MODEL  = "facebook/opt-125m"
TARGET_MODEL = "facebook/opt-1.3b"
 
# Better options (higher acceptance rate, may require A100 or Colab Pro)
DRAFT_MODEL  = "double7/vicuna-68m"        # requires ~14 GB VRAM total
TARGET_MODEL = "lmsys/vicuna-7b-v1.3"
 
# Smaller alternative (fits on T4 with more headroom)
DRAFT_MODEL  = "facebook/opt-350m"
TARGET_MODEL = "facebook/opt-1.3b"
```
 
> Vicuna-7B requires approximately 14.2 GB combined in float16. It fits on a T4 but leaves little headroom. Enable Colab Pro for more reliable sessions at that scale.
 
### Implement full KV-cache shifting
 
The biggest performance gap. In Cell 7, replace the cache reset in Phase 4 with proper δᵢ shifting:
 
```python
# Current implementation (correct, conservative):
# After Phase 4, target_past is set to None and recomputed next round.
 
# Target implementation (full EQSPEC):
# After unpad-append-repad, compute per-sequence offsets:
#   delta_i = (new_L - old_L) - accepted_i
# Then shift KV-cache tensors (shape: batch × heads × seq × dim):
#   for each layer l:
#       new_kv[i] = pad_left(old_kv[i], delta_i)
# This keeps verification cost constant regardless of sequence length.
```
 
This requires working with PyTorch's `past_key_values` tuple structure. A good reference is Section 3.1 of the EQSPEC paper and the padding offset formula in Equation (2).
 
### Implement the EXSPEC executor
 
The `PoolNode` IR node exists and passes the static checker, but its executor is not yet implemented. EXSPEC avoids realignment overhead by grouping same-length sequences across a sliding window:
 
```python
# In Cell 7, add a new executor class:
class EXSPECExecutor(EQSPECExecutor):
    def __init__(self, window_size=16, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
 
    def _run_batch(self, prompts, max_new_tokens):
        # 1. Initialize SequencePool with all prompts
        # 2. Fill sliding window of size W > batch_size
        # 3. For each round:
        #    a. Try to form a batch of same-length sequences (no realign needed)
        #    b. If grouping fails, fall back to unpad-append-repad (EQSPEC path)
        #    c. Write accepted tokens back to pool
        #    d. Refill window
        # Reference: Algorithm 2 in Zhang et al. (2026)
        pass
```
 
### Add structural enforcement of Invariant I2
 
The checker currently catches structural invariants (R1–R4). Invariant I2 (position IDs derived from attention mask) is currently enforced at runtime in `_pos_ids()`. A stronger version would catch models that do not accept `position_ids` as an argument at compile time:
 
```python
# In the static checker, add a new rule R5:
def check_position_id_support(graph, target_model_ref):
    """
    R5: Verify that the target model accepts position_ids.
    Models using learned absolute position embeddings (OPT, GPT-2)
    require explicit position_ids to maintain I2 after repadding.
    RoPE models (LLaMA, Vicuna, Qwen) compute positions internally
    and do not need this check.
    """
    import transformers
    config = transformers.AutoConfig.from_pretrained(target_model_ref)
    model_type = config.model_type
    REQUIRES_EXPLICIT_POS = {"opt", "gpt2", "bloom", "gpt_neo"}
    if model_type in REQUIRES_EXPLICIT_POS:
        print(f"  ⚑ Model type '{model_type}' requires explicit position_ids (I2).")
        print(f"    EQSPECExecutor handles this automatically via _pos_ids().")
```
 
### Extend to larger batch sizes
 
Batch size 8 is where Zhang et al. report their main throughput results. Getting there on a T4 requires reducing the per-sequence KV-cache footprint. Two approaches:
 
```python
# Option A: Use 8-bit quantization to halve model memory
from transformers import BitsAndBytesConfig
 
quant_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)
# This reduces model size from ~3.17 GB to ~1.8 GB, freeing ~1.4 GB for batch KV-cache.
 
# Option B: Reduce max_new_tokens to limit KV-cache growth
MAX_NEW_TOKENS = 20   # instead of 30 or 40
# Each token adds (2 × num_layers × num_heads × head_dim × 2 bytes) per sequence.
# For OPT-1.3B: ~0.5 GB per sequence at 40 tokens → ~0.25 GB at 20 tokens.
```

## Known Issues and Limitations
 
| Issue | Status | Workaround |
|-------|--------|-----------|
| KV-cache not shifted by δᵢ | Open — future work | Full recompute each round (correct, slower) |
| EXSPEC `PoolNode` executor not implemented | Open | Use `RealignNode` (EQSPEC) for now |
| Batch size 8 not tested on T4 | Open | Requires 8-bit quantization or Colab Pro |
| OPT model pair has low acceptance rate | By design | Swap to Vicuna-7B/68M for real speedup |
| Session timeout on long runs | Colab limitation | Run Cell 4 and Cell 8 separately |
 
 
## References
 
| # | Citation |
|---|---------|
| [1] | Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://doi.org/10.48550/arXiv.2211.17192) |
| [2] | R. H. Zhang et al. *Batch Speculative Decoding Done Right.* ICLR 2026 (under review). [arXiv:2510.22876](https://doi.org/10.48550/arXiv.2510.22876) |
| [3] | Z. Wu et al. *TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding.* ACL 2025. [aclanthology](https://aclanthology.org/2025.acl-long.1598/) |
| [4] | X. Liu. *Efficient LLM System with Speculative Decoding.* UC Berkeley EECS PhD Thesis, 2025. [arXiv:2406.14066](https://doi.org/10.48550/arXiv.2406.14066) |
| [5] | H. Qian et al. *BASS: Batched Attention-Optimized Speculative Sampling.* ACL Findings 2024. [aclanthology](https://aclanthology.org/2024.findings-acl.489/) |
