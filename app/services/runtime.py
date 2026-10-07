"""ONNX Runtime session configuration and async threadpool / concurrency controls."""
import asyncio
from typing import Any, Callable
import onnxruntime as ort

from app.config import Settings


def create_session_options(settings: Settings) -> ort.SessionOptions:
    """Configures ONNX Runtime session options tuned for CPU inference."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = settings.ort_intra_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


class RuntimeManager:
    """Manages inference concurrency and non-blocking threadpool dispatch."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)

    async def run_in_pool(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Executes CPU-bound inference in threadpool under semaphore concurrency control."""
        async with self.semaphore:
            return await asyncio.to_thread(func, *args, **kwargs)
