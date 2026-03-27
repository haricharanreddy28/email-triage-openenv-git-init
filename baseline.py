"""
Baseline Inference Script — Email Triage OpenEnv
=================================================
Runs an LLM agent (via OpenAI API) against all 3 tasks
and produces reproducible baseline scores.

Usage:
  python baseline.py                    # runs all 3 tasks
  python baseline.py --task easy        # runs one task

Requires:
  OPENAI_API_KEY environment variable set
"""

import os
import json
import argparse
from openai import OpenAI
from environment import EmailTriageEnvironment, Action

# ── Setup ────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
env = EmailTriageEnvironment()

SYSTEM_PROMPT = """You are an expert email triage agent for a SaaS customer support team.

For each email, you must output a JSON object with exactly these fields:
{
  "priority": "<urgent|normal|low>",
  "category": "<billing|support|spam|inquiry>",
  "route_to": "<billing_team|support_team|trash|sales_team>"
}

Guidelines:
- urgent: needs action within hours (payment issues, account locked, production down, data breach)
- normal: needs action within 1-2 days (general questions, invoices, renewal reminders)
- low: can be addressed this week or ignored (newsletters, feature requests, resolved tickets)

- billing: payment, invoices, charges, refunds, subscriptions
- support: technical issues, bugs, login problems, how-to questions
- spam: unsolicited ads, scam emails, mass marketing
- inquiry: sales questions, partnership requests, pricing, feature requests

- billing_team: all billing/payment emails
- support_team: all technical support emails
- trash: spam and low-value automated emails
- sales_team: sales inquiries, partnership, enterprise deals, pricing questions

Respond with ONLY the JSON object. No explanation. No markdown.
"""

# ── Agent ────────────────────────────────────────

def call_llm(subject: str, body: str, sender: str) -> dict:
    """Call the LLM and parse its JSON response."""
    user_message = f"""
Email to triage:
FROM: {sender}
SUBJECT: {subject}
BODY: {body}

Classify this email now.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap and fast, good enough for baseline
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,    # deterministic = reproducible scores
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()

    # Strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(task_id: str, verbose: bool = True) -> float:
    """Run the baseline agent on a single task and return the score."""
    obs_dict = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'='*60}")

    total_reward = 0.0
    step_count = 0

    while True:
        # Check if done
        current_state = env.state()
        if current_state["done"]:
            break

        # Parse current observation
        subject = obs_dict.get("subject", "")
        body = obs_dict.get("body", "")
        sender = obs_dict.get("sender", "")
        email_id = obs_dict.get("email_id", "")

        if email_id == "DONE":
            break

        # Ask LLM to classify
        try:
            prediction = call_llm(subject, body, sender)
        except Exception as e:
            if verbose:
                print(f"  [Step {step_count+1}] LLM error: {e}. Using fallback.")
            prediction = {"priority": "normal", "category": "inquiry", "route_to": "sales_team"}

        action = Action(
            priority=prediction.get("priority", "normal"),
            category=prediction.get("category", "inquiry"),
            route_to=prediction.get("route_to", "sales_team"),
        )

        obs_dict, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        if verbose:
            correct = "✅" if (info["correct_priority"] and info["correct_category"] and info["correct_route"]) else "⚠️ "
            print(f"  {correct} Email {step_count}: reward={reward:.2f} | "
                  f"priority={action.priority} (expected: {info['label']['priority']}) | "
                  f"category={action.category} (expected: {info['label']['category']})")

        if done:
            break

    final_score = round(total_reward / step_count, 4) if step_count > 0 else 0.0

    if verbose:
        print(f"\n  Final score: {final_score}")
        print(f"  Total emails processed: {step_count}")

    return final_score


# ── Main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Email Triage baseline agent")
    parser.add_argument("--task", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="Which task to run (default: all)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set. Set it with:")
        print("   Windows: set OPENAI_API_KEY=sk-...")
        print("   Mac/Linux: export OPENAI_API_KEY=sk-...")
        return

    tasks_to_run = [args.task] if args.task else ["easy", "medium", "hard"]
    results = {}

    for task_id in tasks_to_run:
        score = run_task(task_id, verbose=True)
        results[task_id] = score

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_id, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} {score:.4f}  {bar}")
    print()


if __name__ == "__main__":
    main()
