"""Module G — the Ollama advisory layer.

Advisory/narrative only: a local LLM annotates already-made autonomous decisions (strategy
adoption, activity-governor widening) with a plain-English rationale. It is never a gate — it
cannot block, veto, or alter a decision — and it fails open, returning `None` on any error so a
down/slow/missing model never affects trading. See `app.ai_advisor.client.narrate`.
"""

from app.ai_advisor.client import narrate

__all__ = ["narrate"]
