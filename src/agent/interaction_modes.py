"""
interaction_modes.py - Modos de interacción del agente

Define cómo el agente responde según la configuración del usuario.
Se inyecta como instrucción adicional en los system prompts de todos los workers.

MODOS DISPONIBLES:
- Mode: Chat / Code / Agent / Voice

USO:
    from src.agent.interaction_modes import get_mode_instructions

    instructions = get_mode_instructions(state)
    prompt = BASE_PROMPT + instructions
"""
from typing import Dict, Any


# ============================================
# MODE: Cómo responde el agente
# ============================================

MODE_INSTRUCTIONS = {
    "chat": "",  # Default - sin instrucciones extra, comportamiento normal

    "code": """
RESPONSE MODE: CODE
- Respond primarily with code. Minimize prose.
- When explaining, use inline comments rather than paragraphs.
- Use code blocks with proper language tags.
- If the user asks a question, answer with a code example first, explanation second.
- Skip greetings and pleasantries — go straight to code.
- Format: ```language\\ncode\\n```
""",

    "agent": """
RESPONSE MODE: AGENT (EXECUTION)
- Be extremely concise. Maximum 1-2 sentences per response.
- Execute actions immediately without asking for confirmation (unless safety-critical).
- Report results in telegram style: "✅ Done: [what happened]" or "❌ Failed: [why]"
- No explanations unless the user asks "why" or "how".
- No greetings, no pleasantries, no suggestions.
- Think of yourself as a CLI tool: input → action → result.
- Examples:
  - User: "move robot X 20" → "✅ Moved X +20mm. Position: (120, 50, 200)"
  - User: "status station 3" → "🟢 Station 3: Online | PLC: OK | Cobot: Idle | Doors: Closed"
  - User: "start cobot 1" → "✅ Cobot started on Station 1, Mode 1"
""",

    "voice": """
RESPONSE MODE: VOICE
- Respond in natural, conversational language optimized for text-to-speech.
- Avoid markdown, bullet points, code blocks, or special formatting.
- Use short, clear sentences. Avoid abbreviations.
- Numbers: say "twenty millimeters" not "20mm".
- Don't use emojis or special characters.
- Structure: one idea per sentence, pause-friendly.
""",
}


# ============================================
# PUBLIC API
# ============================================

def get_mode_instructions(state: Dict[str, Any]) -> str:
    """
    Genera las instrucciones de modo basándose en el state del agente.

    Lee de state:
        - interaction_mode: "chat" | "code" | "agent" | "voice"

    Returns:
        String con instrucciones para inyectar en el system prompt.
        Vacío si todo está en default.
    """
    mode = state.get("interaction_mode", "chat").lower()

    mode_text = MODE_INSTRUCTIONS.get(mode, "")
    if not mode_text:
        return ""

    return "\n\n" + mode_text.strip() + "\n"
