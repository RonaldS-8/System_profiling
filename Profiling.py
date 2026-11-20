from __future__ import annotations

import timeit
import torch
import argparse
import statistics
import math
from einops import einsum
import cs336_basics.model as md
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW 
from cs336_basics.nn_utils import cross_entropy, softmax
from torch import Tensor
from torch.cuda import nvtx


@nvtx.range("scaled_dot_product_attention")
def timed_scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
)-> Tensor:
    """compute the scaled dot product attention and time it"""
    d_k = K.shape[-1]
    with nvtx.range("compute attention_scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("compute softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("compute output"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output
md.scaled_dot_product_attention = timed_scaled_dot_product_attention  

class Benchmarking:
    """a class for benchmarking the transformer model"""
    
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float | None = 10_000.0,
    ):
        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.device.type == "cuda", "GPU is required for benchmarking"
        self.model = BasicsTransformerLM(**self.config).to(self.device)
        self.optimizer = AdamW(self.model.parameters())
        self.model.eval()  # set the model to evaluation mode

    def _generate_random_input(self, batch_size: int) -> Tensor:
        """generate random input for the transformer model"""
        sequence_length = self.config["context_length"]
        input_ids = torch.randint(0, self.config.get("vocab_size"), (batch_size, sequence_length))
        return input_ids

    def benchmark(self, batch_size: int, test_type: str, warmup_steps: int = 5, test_steps: int = 10) -> list[float]:
        """benchmark the transformer model, using test_type to determine whether to benchmark forward or backward or both"""
        input_ids = self._generate_random_input(batch_size).to(self.device)
        result = []
        torch.cuda.synchronize()  # wait for all kernels to finish
        if test_type == "forward":
            for _ in range(warmup_steps):  # run a few iterations to warm up the GPU
                _ = self.model(input_ids)
            torch.cuda.synchronize()
            with nvtx.range("measurement"):
                for _ in range(test_steps):
                    start_time = timeit.default_timer()
                    with nvtx.range("forward"):
                        _ = self.model(input_ids)
                        torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    result.append(end_time - start_time)
        elif test_type == "backward":
            for _ in range(warmup_steps):  # run a few iterations to warm up the GPU
                output = self.model(input_ids)
                loss = cross_entropy(output, input_ids)
                self.optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
            with nvtx.range("measurement"):
                for _ in range(test_steps):
                    start_time = timeit.default_timer()
                    with nvtx.range("forward"):
                        output = self.model(input_ids)
                    with nvtx.range("loss"):
                        loss = cross_entropy(output, input_ids)
                    with nvtx.range("zero_grad"):
                        self.optimizer.zero_grad()
                    with nvtx.range("backward"):
                        loss.backward()
                    torch.cuda.synchronize() 
                    end_time = timeit.default_timer()
                    result.append(end_time - start_time)
        elif test_type == "both":
            for _ in range(warmup_steps):  # run a few iterations to warm up the GPU
                output = self.model(input_ids)
                loss = cross_entropy(output, input_ids)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            torch.cuda.synchronize()
            with nvtx.range("measurement"):
                for _ in range(test_steps):
                    start_time = timeit.default_timer()
                    with nvtx.range("forward"):
                        output = self.model(input_ids)
                    with nvtx.range("loss"):
                        loss = cross_entropy(output, input_ids)
                    with nvtx.range("zero_grad"):
                        self.optimizer.zero_grad()
                    with nvtx.range("backward"):
                        loss.backward()
                    with nvtx.range("optimizer_step"):
                        self.optimizer.step()
                    torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    result.append(end_time - start_time)  
        else:
                raise ValueError(f"Invalid test_type: {test_type}. Must be 'forward', 'backward', or 'both'.")
        return result

    def run_benchmarks(self, batch_sizes: list[int], test_types: list[str], warmup_steps: int = 5, test_steps: int = 10) -> dict:
        """run benchmarks for different batch sizes and test types"""
        results = {}
        for batch_size in batch_sizes:
            for test_type in test_types:
                time_taken = self.benchmark(batch_size, test_type, warmup_steps, test_steps)
                results[(batch_size, test_type)] = time_taken
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Transformer Model")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feedforward dimension")
    parser.add_argument("--rope_theta", type=float, default=10_000.0, help="RoPE theta value")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[4], help="List of batch sizes to benchmark")
    parser.add_argument("--test_types", type=str, choices=["forward", "backward", "both"], default="forward", help="List of test types to benchmark")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--test_steps", type=int, default=10, help="Number of test steps")

    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup_steps must be nonnegative")
    if args.test_steps <= 0:
        parser.error("--test_steps must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch_sizes must contain only positive values")

    benchmarker = Benchmarking(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    )

    results = benchmarker.run_benchmarks(args.batch_sizes, [args.test_types], args.warmup_steps, args.test_steps)
    
    for (batch_size, test_type), times in results.items():
        mean_time = statistics.mean(times)
        std_time = statistics.pstdev(times)
        print(
            f"Batch Size: {batch_size}, Test Type: {test_type}, "
            f"Mean: {mean_time:.6f} s, Standard deviation (population): {std_time:.6f} s, "
            f"Times (s): {times}"
        )