import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.strategy import (
    get_model_strategy,
    TinyLocalStrategy,
    StandardLocalStrategy,
    CloudStrategy
)

class TestAgentStrategy(unittest.TestCase):
    def test_cloud_strategy_selection(self):
        # zai provider -> CloudStrategy
        strategy = get_model_strategy("some-model", "zai")
        self.assertIsInstance(strategy, CloudStrategy)
        self.assertEqual(strategy.get_max_history_messages("cloud"), 80)
        self.assertTrue(strategy.supports_native_tools())

    def test_tiny_model_strategy_selection(self):
        # 1.5b local model -> TinyLocalStrategy
        strategy = get_model_strategy("qwen2.5:1.5b", "ollama")
        self.assertIsInstance(strategy, TinyLocalStrategy)
        self.assertEqual(strategy.get_max_history_messages("tiny"), 12)
        self.assertFalse(strategy.supports_native_tools())

    def test_standard_local_strategy_selection(self):
        # 7b local model -> StandardLocalStrategy
        strategy = get_model_strategy("qwen2.5:7b", "ollama")
        self.assertIsInstance(strategy, StandardLocalStrategy)
        self.assertEqual(strategy.get_max_history_messages("medium"), 40)
        self.assertTrue(strategy.supports_native_tools())

if __name__ == "__main__":
    unittest.main()
