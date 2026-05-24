"""Simulate two full conversations: happy path and failure path."""
from chatbot import AnthemChatbot

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
GRAY  = "\033[90m"
RED   = "\033[91m"


def run_scenario(title: str, turns: list[str]) -> None:
    print(f"\n{'='*64}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{'='*64}\n")

    bot = AnthemChatbot()
    print(f"{GREEN}AI Agent:{RESET} {bot.greeting()}\n")

    for user_input in turns:
        print(f"{CYAN}You:{RESET} {user_input}")
        reply = bot.respond(user_input)
        print(f"{GREEN}AI Agent:{RESET} {reply}\n")
        if bot.ended:
            break


# ── Scenario 1: happy path (correct member ID on first try) ────────────────
run_scenario(
    "Scenario 1 — Happy Path (correct memberID first attempt)",
    [
        "yes",          # Step 1 → confirm facility
        "yes",          # Step 2 → confirm patient
        "0356890AT",    # Step 3 → memberID matches
        "Ready",        # Step 6 → ready to receive info
        "No",           # Step 7 → no repeat needed
        "Thank you",    # Step 8 → farewell
    ],
)

# ── Scenario 2: memberID wrong twice, correct on 3rd attempt ──────────────
run_scenario(
    "Scenario 2 — MemberID Wrong Twice, Correct on 3rd Attempt",
    [
        "yes",          # Step 1
        "yes",          # Step 2
        "1234567XX",    # Step 3 → wrong
        "9999999ZZ",    # Step 4 → wrong again
        "0356890AT",    # Step 5 → correct on final attempt
        "Ready",        # Step 6
        "Yes, repeat",  # Step 7 → wants repeat
        "No",           # Step 7 again → done
        "Bye",          # Step 8 → farewell
    ],
)

# ── Scenario 3: memberID fails all 3 attempts → Step 10 ───────────────────
run_scenario(
    "Scenario 3 — MemberID Verification Fails (3 wrong attempts)",
    [
        "yes",          # Step 1
        "yes",          # Step 2
        "WRONG111",     # Step 3 → wrong
        "WRONG222",     # Step 4 → wrong
        "WRONG333",     # Step 5 → wrong, locked out
        "Thank you",    # Step 10 → farewell
    ],
)

# ── Scenario 4: wrong facility → Step 9 ───────────────────────────────────
run_scenario(
    "Scenario 4 — Wrong Facility (Step 9 early exit)",
    [
        "No",           # Step 1 → wrong person
        "Bye",          # Step 9 → farewell
    ],
)
