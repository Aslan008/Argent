"""
Argent MicroModels Package.

Provides ultra-fast in-process micromodels for intent routing and argument extraction.
"""

from src.micromodels.data_collector import MicroDataCollector, collector
from src.micromodels.dispatcher import FastDispatchResult, MicroModelDispatcher, dispatcher
from src.micromodels.intent_router import TinyIntentEncoder
from src.micromodels.runtime import MicroModelRuntime, runtime
from src.micromodels.tokenizer import BPETokenizer

__all__ = [
    "BPETokenizer",
    "TinyIntentEncoder",
    "MicroModelRuntime",
    "runtime",
    "MicroModelDispatcher",
    "dispatcher",
    "MicroDataCollector",
    "collector",
    "FastDispatchResult",
]
