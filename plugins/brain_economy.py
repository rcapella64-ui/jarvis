"""
Brain — settings form and a status tool.

This plugin exists for its PLUGIN_SETTINGS: it puts the chat provider, key and
model into ⚙ → PLUGIN SETTINGS → BRAIN, with a TEST CONNECTION button. The
brain itself lives in core/brain.py. It needs Groq listening for the ears
(⚙ → PLUGIN SETTINGS → GROQ). The voice is not a setting: the assistant always
speaks with the clone of voz_local/referencia.wav.
"""
from core.brain import NAMESPACE, PROVIDERS, test_connection, voice_label

PLUGIN = {
    "name": "brain_status",
    "description": (
        "Reports which chat model is writing the replies (Gemini text by default, "
        "or DeepSeek, Groq, OpenRouter…) and the voice reading them aloud. Use when "
        "the user asks what model or AI is answering."
    ),
    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
}

PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "BRAIN (Gemini / DeepSeek / Groq chat model)",
    "fields": [
        {"key": "provider", "label": "Provider", "type": "choice",
         "options": list(PROVIDERS), "default": "gemini"},
        {"key": "api_key", "label": "API key (blank = the Gemini key from setup)",
         "type": "password", "placeholder": "AIza... (Gemini) · sk-... (DeepSeek) · gsk_... (Groq)"},
        {"key": "model", "label": "Model (blank = provider default)", "type": "text",
         "placeholder": "gemini-3.6-flash · gemini-3.5-flash-lite · deepseek-chat"},
        {"key": "base_url", "label": "Base URL (custom provider only)", "type": "text",
         "placeholder": "http://localhost:11434/v1"},
    ],
    "action": {"label": "TEST CONNECTION", "run": test_connection},
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    from memory.config_manager import get_plugin_config
    from core.brain import settings
    st = settings(get_plugin_config)
    return (f"The replies are written by {st['model']} via {st['provider']} and read "
            f"aloud by {voice_label(st)}.")
