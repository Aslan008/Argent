import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import ArgentAgent
from src.agent.trimmer import estimate_tokens, clean_messages_for_llm, sliding_window_trim
from tools._helpers import _validate_code_syntax


class TestArgentEnhancements(unittest.TestCase):

    @patch('agent.get_obsidian_vault')
    @patch('agent.get_current_model')
    @patch('agent.get_model_size_category')
    def test_dynamic_prompt_compilation_cloud(self, mock_category, mock_model, mock_vault):
        # Setup mock behavior for large cloud model
        mock_model.return_value = "gpt-4"
        mock_category.return_value = "large"
        mock_vault.return_value = "C:\\path\\to\\my\\vault"

        agent = ArgentAgent()
        prompt = agent.build_system_prompt()

        # Check standard sections
        self.assertIn("CRITICAL: LANGUAGE RULE", prompt)
        self.assertIn("ROLE: Argent Coder", prompt)
        self.assertIn("## 4. PLANNING MODE & ARTIFACTS", prompt)

    @patch('agent.get_obsidian_vault')
    @patch('agent.get_current_model')
    @patch('agent.get_model_size_category')
    def test_dynamic_prompt_compilation_tiny_no_vault(self, mock_category, mock_model, mock_vault):
        # Setup mock behavior for tiny local model without Obsidian vault
        mock_model.return_value = "qwen2.5:1.5b"
        mock_category.return_value = "tiny"
        mock_vault.return_value = ""

        agent = ArgentAgent()
        prompt = agent.build_system_prompt()

        self.assertIn("CRITICAL: LANGUAGE RULE", prompt)
        self.assertIn("ROLE: Argent Coder", prompt)
        # Should NOT have Obsidian section
        self.assertNotIn("OBSIDIAN INTEGRATION", prompt)
        # Tiny model should NOT have Project Brain Mode instructions
        self.assertNotIn("PROJECT BRAIN MODE", prompt)
        # Tiny model should have a shorter/simplified operational protocol
        self.assertIn("- **Tool-First**: Invoke tools immediately via JSON when needed.", prompt)

    def test_token_estimation_with_tiktoken(self):
        # If tiktoken is installed, we should verify it runs and is correct.
        # Otherwise, the heuristic fallback should be active.
        test_text = "Hello world! This is a test string."
        tokens = estimate_tokens(test_text, "gpt-4", "zai")
        self.assertGreater(tokens, 0)

        # Test Cyrillic heuristic
        ru_text = "Привет мир! Это тестовая строка."
        tokens_ru = estimate_tokens(ru_text, "qwen2.5:7b", "ollama")
        self.assertGreater(tokens_ru, 0)

    @patch('tools._helpers.py_compile.compile')
    def test_python_syntax_validation(self, mock_compile):
        # Valid Python
        mock_compile.return_value = True
        err = _validate_code_syntax("test_file.py")
        self.assertIsNone(err)

        # Invalid Python
        import py_compile
        mock_compile.side_effect = py_compile.PyCompileError(SyntaxError, SyntaxError("invalid syntax"), "test_file.py", "SyntaxError: invalid syntax")
        err = _validate_code_syntax("test_file.py")
        self.assertIsNotNone(err)
        self.assertIn("SyntaxError in your Python code", err)

    def test_json_syntax_validation(self):
        # Setup a temp valid json file
        valid_json_path = "temp_valid_test.json"
        with open(valid_json_path, "w", encoding="utf-8") as f:
            json.dump({"key": "value", "number": 123}, f)

        # Setup an invalid json file
        invalid_json_path = "temp_invalid_test.json"
        with open(invalid_json_path, "w", encoding="utf-8") as f:
            f.write("{'invalid_key': value,}")

        try:
            # Valid json file check
            err_valid = _validate_code_syntax(valid_json_path)
            self.assertIsNone(err_valid)

            # Invalid json file check
            err_invalid = _validate_code_syntax(invalid_json_path)
            self.assertIsNotNone(err_invalid)
            self.assertIn("JSON Syntax Error", err_invalid)
        finally:
            # Cleanup temp files
            if os.path.exists(valid_json_path):
                os.remove(valid_json_path)
            if os.path.exists(invalid_json_path):
                os.remove(invalid_json_path)

    # Patched at run_text, not subprocess.run: run_text now drives Popen itself
    # so it can kill a whole process tree on timeout, and a patch one layer too
    # low silently stopped intercepting — the real `node --check` ran.
    @patch('tools._helpers.run_text')
    def test_javascript_syntax_validation(self, mock_run):
        # Valid JS (returncode 0)
        mock_run.return_value = MagicMock(returncode=0)
        err = _validate_code_syntax("script.js")
        self.assertIsNone(err)

        # Invalid JS (returncode 1)
        mock_run.return_value = MagicMock(returncode=1, stderr="SyntaxError: Unexpected token", stdout="")
        err = _validate_code_syntax("script.js")
        self.assertIsNotNone(err)
        self.assertIn("JavaScript Syntax Error", err)

    def test_clean_messages_for_llm_strip_enabled(self):
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Let me think.<think>nested reasoning</think>\nHere is the answer.", "thinking": "raw reasoning"}
        ]
        cleaned = clean_messages_for_llm(msgs, strip_reasoning=True)
        
        self.assertEqual(len(cleaned), 3)
        self.assertNotIn("thinking", cleaned[2])
        # Both the 'thinking' key and the <think> tag inside content should be stripped
        self.assertEqual(cleaned[2]["content"], "Let me think.\nHere is the answer.")

    def test_clean_messages_for_llm_strip_disabled(self):
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Here is the answer.", "thinking": "raw reasoning"}
        ]
        cleaned = clean_messages_for_llm(msgs, strip_reasoning=False)
        
        self.assertEqual(len(cleaned), 3)
        self.assertNotIn("thinking", cleaned[2])
        # The reasoning from 'thinking' should be formatted back into content
        self.assertIn("<think>\nraw reasoning\n</think>", cleaned[2]["content"])
        self.assertIn("Here is the answer.", cleaned[2]["content"])

    @patch('src.agent.trimmer.get_provider')
    @patch('src.agent.trimmer.get_strip_reasoning')
    def test_sliding_window_trim(self, mock_strip, mock_provider):
        mock_strip.return_value = True
        mock_provider.return_value = "zai"
        
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user turn 1"},
            {"role": "assistant", "content": "assistant turn 1"},
            {"role": "user", "content": "user turn 2"},
            {"role": "assistant", "content": "assistant turn 2"},
            {"role": "user", "content": "user turn 3"},
        ]
        
        # Slicing with max_context_tokens set small should trigger trimming
        trimmed = sliding_window_trim(msgs, "gpt-4", max_history_messages=10, max_context_tokens=30)
        
        # Verify system prompt (index 0) is kept
        self.assertEqual(trimmed[0]["role"], "system")
        # Verify it sliced at a user boundary
        self.assertEqual(trimmed[1]["role"], "user")
        # Verify the structure is preserved
        self.assertTrue(len(trimmed) < len(msgs))

    def test_temperature_configuration(self):
        from config import get_temperature, set_temperature
        original = get_temperature()
        try:
            set_temperature(0.45)
            self.assertEqual(get_temperature(), 0.45)
        finally:
            set_temperature(original)

    def test_ollama_provider_temperature(self):
        from providers import OllamaProvider
        provider = OllamaProvider()
        mock_ollama = MagicMock()
        provider._ollama = mock_ollama
        mock_ollama.chat.return_value = []
        
        # Consume generator
        list(provider.stream_chat("llama3", [{"role": "user", "content": "hi"}], temperature=0.6))
        
        mock_ollama.chat.assert_called_once()
        called_kwargs = mock_ollama.chat.call_args[1]
        self.assertIn("options", called_kwargs)
        self.assertEqual(called_kwargs["options"].get("temperature"), 0.6)

    def test_openai_compatible_provider_temperature(self):
        from providers import ZAIProvider
        provider = ZAIProvider(api_key="mock_key", base_url="http://mock_url")
        
        mock_client = MagicMock()
        provider._get_client = MagicMock(return_value=mock_client)
        
        # Consume generator
        list(provider.stream_chat("gpt-4", [{"role": "user", "content": "hi"}], temperature=0.85))
        
        mock_client.chat.completions.create.assert_called_once()
        called_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(called_kwargs.get("temperature"), 0.85)


if __name__ == "__main__":
    unittest.main()
