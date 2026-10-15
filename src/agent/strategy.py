from abc import ABC, abstractmethod
from typing import List, Dict, Any
import config
from config import get_model_size_category, is_moe_model, get_model_category_override
from src.agent.parser import parse_raw_tool_call
from src.agent.trimmer import hard_reset_with_memory, soft_trim_with_summarization, sliding_window_trim

class ModelStrategy(ABC):
    """Abstract base class representing strategy optimizations for a specific LLM tier."""

    @abstractmethod
    def get_max_history_messages(self, category: str) -> int:
        pass

    @abstractmethod
    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def parse_response(self, content: str) -> Dict[str, Any] | None:
        pass

    @abstractmethod
    def supports_native_tools(self) -> bool:
        pass

    def wants_constrained_decoding(self) -> bool:
        """Whether agent steps should be grammar-constrained to the step JSON
        schema (only effective when the provider supports it)."""
        return False

    def wants_objective_anchor(self) -> bool:
        """Whether a trailing OBJECTIVE reminder should be appended to long
        contexts: small models attend to the end of the prompt and lose the
        original goal as history grows."""
        return False

    def wants_factual_anchor_only(self) -> bool:
        """Whether the objective anchor should include ONLY ground-truth facts (objective + touched files),
        omitting speculative failure directives and subtasks."""
        return False

    def wants_minified_schemas(self) -> bool:
        """Whether tool schemas should be minified to save tokens and avoid router distraction."""
        return False

    def summarizes_history(self) -> bool:
        """Whether trimming writes a model summary of what it drops (so one can be written ahead of time)."""
        return False

    def is_loop_guard_enabled(self) -> bool:
        """Whether LoopGuard should monitor and intercept tool loops."""
        return True

    def is_auto_healing_enabled(self) -> bool:
        """Whether failed tool calls should inject structured auto-healing instructions."""
        return True

    def evaluate_command_safety(self, command: str) -> tuple[bool, float, str]:
        """Evaluates whether a command is safe to execute using Leeth System One model.

        Returns:
            (is_safe: bool, confidence: float, reason: str)
        """
        if not config.get_leeth_enabled():
            return True, 1.0, "Leeth disabled"
        try:
            from src.leeth.runtime import leeth
            from src.leeth.primitives import ChoiceQuestion, NoulQuestion, ScoreQuestion
            if not leeth.is_available:
                return True, 1.0, "Leeth model not available"

            state = f"User shell execution request:\nCommand: {command}\nCaller: Argent Agent"
            questions = [
                NoulQuestion(
                    id="is_safe",
                    question="Является ли данная терминальная команда безопасной для выполнения?",
                ),
                ScoreQuestion(
                    id="risk_level",
                    question="Оцените уровень риска выполнения данной команды от 0 (полностью безопасно) до 100 (катастрофический риск)",
                    rubric=["Безопасно", "Низкий", "Умеренный", "Опасный", "Катастрофический"],
                ),
                ChoiceQuestion(
                    id="action",
                    question="Какое действие должен предпринять агент перед запуском этой команды?",
                    choices=["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"],
                ),
            ]
            batch = leeth.decide(state, questions)
            safe_res = batch.results.get("is_safe")
            score_res = batch.results.get("risk_level")
            action_res = batch.results.get("action")

            is_safe = safe_res.value if safe_res else True
            confidence = safe_res.probability if safe_res else 1.0
            label = score_res.label if score_res else "Unknown"

            if not is_safe or (score_res and score_res.level >= 4) or (action_res and action_res.choice == "block_command"):
                return False, confidence, f"Leeth flagged command as dangerous ({label}, confidence {confidence:.2f})"
            return True, confidence, f"Leeth verified command safety ({label})"
        except Exception as e:
            return True, 1.0, f"Leeth safety check error: {e}"

    def analyze_compiler_error(self, output: str) -> dict:
        """Analyzes a compilation or runtime error using Leeth System One model.

        Extracts language, target file, severity, and suggested recovery tool.
        """
        if not config.get_leeth_enabled():
            return {}
        try:
            from src.leeth.runtime import leeth
            from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion, ScoreQuestion
            if not leeth.is_available:
                return {}

            questions = [
                NoulQuestion(
                    id="has_error",
                    question="Содержит ли вывод компилятора критическую ошибку?",
                ),
                ChoiceQuestion(
                    id="language",
                    question="Какой язык программирования используется в данном компоненте?",
                    choices=["C++", "C# (.NET)", "Rust", "Python", "JavaScript"],
                ),
                ChoiceQuestion(
                    id="suggested_tool",
                    question="Какой инструмент Argent оптимален для исправления данной проблемы?",
                    choices=["read_file", "replace_in_file", "search_files", "run_command", "delete_file"],
                ),
                ExtractQuestion(
                    id="file_path",
                    question="Укажите путь к проблемному файлу C++",
                ),
                ScoreQuestion(
                    id="urgency",
                    question="Оцените критичность и приоритет исправления данной проблемы по шкале 0..100",
                    rubric=["Низкий", "Средний", "Высокий", "Критический", "Блокирующий"],
                ),
            ]
            batch = leeth.decide(output, questions)
            has_err = batch.results.get("has_error")
            lang = batch.results.get("language")
            tool = batch.results.get("suggested_tool")
            file_p = batch.results.get("file_path")
            sev = batch.results.get("urgency")

            return {
                "has_error": has_err.value if has_err else False,
                "error_probability": has_err.probability if has_err else 0.0,
                "language": lang.choice if lang else "Other",
                "suggested_tool": tool.choice if tool else "read_file",
                "target_file": file_p.text if file_p else "",
                "severity_level": sev.level if sev else 1,
                "severity_label": sev.label if sev else "Низкий",
                "elapsed_ms": batch.elapsed_ms,
            }
        except Exception:
            return {}
            return {}



class TinyLocalStrategy(ModelStrategy):
    """Optimization strategy for extremely small local models (< 3B)."""

    def get_max_history_messages(self, category: str) -> int:
        return 12

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        # For extremely small models, keep sliding window or fallback
        return sliding_window_trim(messages, model_name, max_history_messages, max_context_tokens)

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        # Fallback to parsing raw tool call JSON
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return False

    def wants_constrained_decoding(self) -> bool:
        return True

    def wants_objective_anchor(self) -> bool:
        return True

    def wants_minified_schemas(self) -> bool:
        return True


class StandardLocalStrategy(ModelStrategy):
    """Optimization strategy for standard local models (3B - 14B, e.g. Qwen2.5-Coder)."""

    def get_max_history_messages(self, category: str) -> int:
        if category == "small":
            return 20
        return 40

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        # Use Dynamic Summarization for 14-20b models to compress context without losing key facts
        return soft_trim_with_summarization(messages, model_name, max_history_messages, max_context_tokens)

    def summarizes_history(self) -> bool:
        return True

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return True

    def wants_objective_anchor(self) -> bool:
        return True

    def is_loop_guard_enabled(self) -> bool:
        return True

    def is_auto_healing_enabled(self) -> bool:
        return True


class MoELocalStrategy(ModelStrategy):
    """Optimization strategy for local Mixture of Experts models (e.g. ORNITH1.5:35A3B, Mixtral)."""

    def get_max_history_messages(self, category: str) -> int:
        return 28

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        return soft_trim_with_summarization(messages, model_name, max_history_messages, max_context_tokens)

    def summarizes_history(self) -> bool:
        return True

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return True

    def wants_objective_anchor(self) -> bool:
        return True

    def wants_factual_anchor_only(self) -> bool:
        return True

    def wants_minified_schemas(self) -> bool:
        return True

    def is_loop_guard_enabled(self) -> bool:
        return True

    def is_auto_healing_enabled(self) -> bool:
        return False


class CloudStrategy(ModelStrategy):
    """Strategy for large cloud models (Gemini, Z.AI, GPT) with large context windows."""

    def get_max_history_messages(self, category: str) -> int:
        if category == "medium":
            return 40
        elif category == "large":
            return 60
        return 80

    def trim_history(self, messages: List[Dict[str, Any]], model_name: str, 
                     max_history_messages: int, max_context_tokens: int) -> List[Dict[str, Any]]:
        # Cloud models handle parallel summarization API calls cleanly. Soft trim keeps detail.
        return soft_trim_with_summarization(messages, model_name, max_history_messages, max_context_tokens)

    def summarizes_history(self) -> bool:
        return True

    def parse_response(self, content: str) -> Dict[str, Any] | None:
        return parse_raw_tool_call(content)

    def supports_native_tools(self) -> bool:
        return True

    def is_loop_guard_enabled(self) -> bool:
        return True

    def is_auto_healing_enabled(self) -> bool:
        return False


def get_model_strategy(model_name: str, provider: str) -> ModelStrategy:
    """Factory to retrieve the appropriate strategy for the active model and provider.

    Priority 1: User's manual category override (wins unconditionally).
    Priority 2: Cloud providers or cloud model tier.
    Priority 3: MoE architecture detection.
    Priority 4: Size-tier category fallback.
    """
    override = config.get_model_category_override()
    if override:
        if override == "cloud":
            return CloudStrategy()
        elif override == "moe":
            return MoELocalStrategy()
        elif override == "tiny":
            return TinyLocalStrategy()
        else:
            return StandardLocalStrategy()

    category = config.get_model_size_category(model_name)
    if provider in ("zai", "openrouter", "claude", "anthropic", "deepseek", "xkiro", "wormsoft") or category == "cloud":
        return CloudStrategy()
    elif category == "moe" or config.is_moe_model(model_name):
        return MoELocalStrategy()
    elif category == "tiny":
        return TinyLocalStrategy()
    else:
        return StandardLocalStrategy()
