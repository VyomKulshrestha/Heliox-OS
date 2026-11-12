"""Conversation eval: scripted conversations through the real fast layer (see __main__.py)."""

# What a paid run copies from the person's [conversation] settings (pilot.real_folders.apply_owner_settings, only=): the
# fast model for each provider and the reply length. Nothing else of that section: not share_screen_text, which a pack
# sets itself; not provisional_answers, which would switch a feature on in the middle of a measurement; not enabled or
# search_url. A setting added to [conversation] later is not copied until it is named here.
OWNER_CONVERSATION_FIELDS = (
    "conversation.claude_model",
    "conversation.openai_model",
    "conversation.openrouter_model",
    "conversation.gemini_model",
    "conversation.meta_model",
    "conversation.ollama_model",
    "conversation.max_reply_tokens",
)
