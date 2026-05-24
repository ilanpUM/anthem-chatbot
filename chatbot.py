import json
import os
import re
from enum import Enum, auto
from pathlib import Path

import anthropic

# Load .env from the same directory as this file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Member data — swap this out for a real DB lookup in production
# ---------------------------------------------------------------------------
MEMBER_DATA = {
    "facility":   "ABC MEDICAL CENTER",
    "first_name": "John",
    "last_name":  "Smith",
    "dob":        "01/15/1985",
    "member_id":  "0356890AT",
    "auth_number": "UM23456890XTD",
    "cpt_code":   "90Q25",
    "stay_from":  "25/05/2026",
    "stay_to":    "26/05/2026",
}

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# LLM — used ONLY to parse free-text input into structured intents
# ---------------------------------------------------------------------------
_INTENT_DESCRIPTIONS = {
    "yes":       "affirmative (yes, yeah, correct, sure, uh-huh, that's right, etc.)",
    "no":        "negative (no, nope, not really, incorrect, wrong, etc.)",
    "ready":     "user is ready to proceed (ready, go ahead, ok, okay, I'm ready, proceed, etc.)",
    "repeat":    "user wants information repeated (repeat, say again, can you repeat, please repeat, etc.)",
    "no_repeat": "user does NOT want repetition (no, that's fine, I got it, no thank you, nope, etc.)",
    "end":       "farewell or closing (thank you, thanks, bye, goodbye, take care, etc.)",
    "member_id": "extract the alphanumeric member ID spoken by the caller (e.g. 0356890AT)",
}


def _strip_code_block(text: str) -> str:
    """Remove markdown code fences the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_json_intent(raw: str, expected_intents: list[str]) -> dict:
    """Parse LLM response into {intent, value}, tolerating varied key names."""
    data = json.loads(raw)
    # Normalise intent key — model sometimes uses 'classification', 'category', etc.
    intent = (
        data.get("intent")
        or data.get("classification")
        or data.get("category")
        or data.get("type")
        or ""
    ).lower()
    # Normalise value key
    value = (
        data.get("value")
        or data.get("extracted")
        or data.get("member_id")
        or data.get("input")
        or data.get("id")
    )
    # Confirm intent is one we expect; default to last in list if not
    if intent not in expected_intents:
        intent = expected_intents[-1]
    return {"intent": intent, "value": value}


def parse_input(user_input: str, expected_intents: list[str]) -> dict:
    """Call Claude Haiku to classify/extract intent — the only LLM usage in the app."""
    options = "\n".join(
        f"- {k}: {_INTENT_DESCRIPTIONS[k]}"
        for k in expected_intents
        if k in _INTENT_DESCRIPTIONS
    )
    intent_list = ", ".join(f'"{i}"' for i in expected_intents)
    prompt = (
        f'Classify this phone-call response. Intent must be one of: {intent_list}\n\n'
        f"Descriptions:\n{options}\n\n"
        f'Caller said: "{user_input}"\n\n'
        f'Rules:\n'
        f'- Use key "intent" for the chosen intent name (exact string from the list above).\n'
        f'- Use key "value" for any extracted data (member ID string), or null.\n'
        f'- If intent is "member_id", put the extracted alphanumeric ID (no spaces/dashes) in "value".\n'
        f'- Reply with raw JSON only — no markdown, no code fences, no extra text.\n'
        f'Example: {{"intent": "yes", "value": null}}'
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _strip_code_block(resp.content[0].text)
    try:
        return _extract_json_intent(raw, expected_intents)
    except (json.JSONDecodeError, KeyError, AttributeError):
        # Regex fallback: for member_id try to pull an alphanumeric token
        if "member_id" in expected_intents:
            match = re.search(r"[A-Z0-9]{6,}", user_input.upper())
            if match:
                return {"intent": "member_id", "value": match.group(0)}
        # For yes/no, do a simple keyword check
        lower = user_input.lower()
        for intent in expected_intents:
            if intent in lower or (intent == "yes" and any(w in lower for w in ("yeah","yep","sure","ok","correct"))):
                return {"intent": intent, "value": None}
        return {"intent": expected_intents[0], "value": None}


# ---------------------------------------------------------------------------
# Question detection and answering
# ---------------------------------------------------------------------------
_QUESTION_STARTERS = {
    "what", "where", "who", "why", "how", "which", "whose", "whom",
    "when", "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "can ", "could ", "would ", "will ", "should ",
}

def _looks_like_question(text: str) -> bool:
    lo = text.lower().strip()
    return lo.endswith("?") or any(
        lo.startswith(w) for w in _QUESTION_STARTERS
    )


def _answer_question(user_input: str) -> str | None:
    """Answer any question the caller asks. Returns None for non-questions."""
    if not _looks_like_question(user_input):
        return None
    m = MEMBER_DATA
    context = (
        "You are an AI virtual assistant calling on behalf of Anthem Blue Cross and Blue Shield. "
        f"You are calling {m['facility']} with an authorization update for patient "
        f"{m['first_name']} {m['last_name']} (DOB {m['dob']}, member ID {m['member_id']}). "
        f"The authorization number is {m['auth_number']}, CPT code {m['cpt_code']}, "
        f"length of stay {m['stay_from']} to {m['stay_to']}."
    )
    prompt = (
        f"{context}\n\n"
        f'The person you called asked: "{user_input}"\n\n'
        "Answer their question briefly and professionally in 1-2 sentences. "
        "If it cannot be answered from the context above, say you can only assist with "
        "this authorization notification. "
        "Reply with just the answer — no preamble, no JSON."
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Deterministic conversation state machine
# ---------------------------------------------------------------------------
class Step(Enum):
    S1  = auto()   # confirm facility
    S2  = auto()   # confirm patient
    S3  = auto()   # memberID attempt 1
    S4  = auto()   # memberID attempt 2
    S5  = auto()   # memberID attempt 3
    S6  = auto()   # authorization approved — wait for "ready"
    S7  = auto()   # share auth details — offer repeat
    S8  = auto()   # closing disclaimer (success path)
    S9  = auto()   # wrong person
    S10 = auto()   # verification failed
    END = auto()


class AnthemChatbot:
    def __init__(self):
        self.step = Step.S1

    @property
    def ended(self) -> bool:
        return self.step == Step.END

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def greeting(self) -> str:
        f = MEMBER_DATA["facility"]
        return (
            "Hello, this is an AI virtual assistant calling from Anthem Blue Cross and Blue Shield. "
            "Just so you know, this call may be monitored or recorded for quality and training purposes. "
            f"Am I speaking with {f}?"
        )

    def respond(self, user_input: str) -> str:
        # Answer any question at any step, then re-ask current prompt
        answer = _answer_question(user_input)
        if answer:
            reprompt = self._reprompt()
            return f"{answer} {reprompt}".strip()

        dispatch = {
            Step.S1:  self._s1,
            Step.S2:  self._s2,
            Step.S3:  self._s3,
            Step.S4:  self._s4,
            Step.S5:  self._s5,
            Step.S6:  self._s6,
            Step.S7:  self._s7,
            Step.S8:  self._farewell,
            Step.S9:  self._farewell,
            Step.S10: self._farewell,
        }
        return dispatch[self.step](user_input)

    def _reprompt(self) -> str:
        """Return the current step's question so conversation continues after a side-question."""
        m = MEMBER_DATA
        name = f'{m["first_name"]} {m["last_name"]}'
        prompts = {
            Step.S1:  f'Am I speaking with {m["facility"]}?',
            Step.S2:  f"Is {name} one of your current patients?",
            Step.S3:  f"Please clearly state the member ID for {name}.",
            Step.S4:  f"Please state slowly the member ID for {name}.",
            Step.S5:  f"Please state the member ID for {name}.",
            Step.S6:  "Let me know when you are ready and I'll share the authorization details.",
            Step.S7:  "Do you want me to repeat any of the authorization details?",
        }
        return prompts.get(self.step, "")

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------
    def _s1(self, user_input: str) -> str:
        r = parse_input(user_input, ["yes", "no"])
        if r["intent"] == "yes":
            self.step = Step.S2
            m = MEMBER_DATA
            return (
                f'I have an important update to share about a decision on '
                f'{m["first_name"]} {m["last_name"]} with date of birth {m["dob"]}. '
                f"Is that one of your current patients or are you familiar with their care?"
            )
        self.step = Step.S9
        return "Looks like I have reached the wrong person. Sorry for the inconvenience. Thank you"

    def _s2(self, user_input: str) -> str:
        r = parse_input(user_input, ["yes", "no"])
        if r["intent"] == "yes":
            self.step = Step.S3
            m = MEMBER_DATA
            return (
                f"To share health details, I need to verify the details. "
                f'Clearly state the memberID for {m["first_name"]} {m["last_name"]}?'
            )
        self.step = Step.S9
        return "Looks like I have reached the wrong person. Sorry for the inconvenience. Thank you"

    def _s3(self, user_input: str) -> str:
        _system_log("Verifying memberID — attempt 1")
        if self._check_member_id(user_input):
            self.step = Step.S6
            return self._s6_prompt()
        self.step = Step.S4
        return self._mismatch_msg()

    def _s4(self, user_input: str) -> str:
        _system_log("Verifying memberID — attempt 2")
        if self._check_member_id(user_input):
            self.step = Step.S6
            return self._s6_prompt()
        self.step = Step.S5
        return self._mismatch_msg()

    def _s5(self, user_input: str) -> str:
        _system_log("Verifying memberID — attempt 3")
        if self._check_member_id(user_input):
            self.step = Step.S6
            return self._s6_prompt()
        self.step = Step.S10
        return "I'm unable to proceed further as the memberID verification has failed. Thank you"

    def _s6(self, user_input: str) -> str:
        r = parse_input(user_input, ["ready"])
        if r["intent"] == "ready":
            self.step = Step.S7
            m = MEMBER_DATA
            return (
                f'For {m["auth_number"]} CPT {m["cpt_code"]} has been approved. '
                f'The length of stay is from {m["stay_from"]} to {m["stay_to"]}. '
                f"Do you want me to repeat any of this info?"
            )
        return "I'll wait until you're ready. Please say 'Ready' when you'd like me to continue."

    def _s7(self, user_input: str) -> str:
        r = parse_input(user_input, ["repeat", "no_repeat"])
        if r["intent"] == "no_repeat":
            self.step = Step.S8
            return (
                "Thank you again for your feedback and your time today. "
                "This is not an approval for claim payment. "
                "This authorization is a confirmation of medical necessity only. "
                "Coverage for this service is subject to: the accuracy of the information we received; "
                "the member's eligibility under the health benefits plan when the service is rendered; "
                "the terms, conditions, limitations and exclusions of the member's health plan, "
                "including any benefit maximums. Have a wonderful day."
            )
        # Repeat the auth details
        m = MEMBER_DATA
        return (
            f'For {m["auth_number"]} CPT {m["cpt_code"]} has been approved. '
            f'The length of stay is from {m["stay_from"]} to {m["stay_to"]}. '
            f"Do you want me to repeat any of this info?"
        )

    def _farewell(self, user_input: str) -> str:
        r = parse_input(user_input, ["end"])
        if r["intent"] == "end":
            self.step = Step.END
            return "Goodbye!"
        return "Is there anything else I can assist you with before we end the call?"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_member_id(self, user_input: str) -> bool:
        result = parse_input(user_input, ["member_id"])
        captured = (result.get("value") or "").strip().upper().replace(" ", "").replace("-", "")
        actual = MEMBER_DATA["member_id"].upper()
        _system_log(f"captured='{captured}'  expected='{actual}'  match={captured == actual}")
        return captured == actual

    def _mismatch_msg(self) -> str:
        m = MEMBER_DATA
        return (
            f"That memberID did not match. "
            f'Please state slowly the memberID for {m["first_name"]} {m["last_name"]}?'
        )

    def _s6_prompt(self) -> str:
        m = MEMBER_DATA
        return (
            f'Great news! I am glad to share that an authorization request for '
            f'{m["first_name"]} {m["last_name"]} has been approved. '
            f"I have some details for you, take a moment to grab a pen and I will share the "
            f"information with you. Let me know when you are ready!"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _system_log(msg: str) -> None:
    print(f"  \033[90m[System: {msg}]\033[0m")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 62)
    print("   Anthem Blue Cross Blue Shield — AI Virtual Assistant")
    print("=" * 62 + "\n")

    bot = AnthemChatbot()
    print(f"AI Agent: {bot.greeting()}\n")

    while not bot.ended:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[Call disconnected]")
            break

        if not user_input:
            continue

        reply = bot.respond(user_input)
        print(f"\nAI Agent: {reply}\n")


if __name__ == "__main__":
    main()
