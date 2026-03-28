"""
Baseline Inference Script - Email Triage OpenEnv
=================================================
Uses FREE Google Gemini API.

Get your FREE Gemini API key at:
  https://aistudio.google.com/apikey

Usage:
  python baseline.py                 # runs all 3 tasks
  python baseline.py --task easy     # runs one task

Requires:
  pip install google-genai
  set GEMINI_API_KEY=your-key-here
"""

import os
import json
import argparse
from google import genai
from google.genai import types
from environment import EmailTriageEnvironment, Action

# ── Setup ─────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)
env = EmailTriageEnvironment()

PROMPT_TEMPLATE = """You are an expert email triage agent. Classify each email into priority, category, and routing.

PRIORITY RULES (pick exactly one):
- "urgent"  = needs response within hours. Use for: payment failed, charged twice, account suspended/locked, production down, API down, data breach, refund request, wrong bill amount, large enterprise inquiry (500+ users)
- "normal"  = needs response within 1-2 days. Use for: general how-to questions, invoice received, subscription renewal reminder, partnership inquiry (not large enterprise), export data questions
- "low"     = can wait or ignore. Use for: spam, newsletters, feature requests, student discount requests, resolved tickets, automated emails

CATEGORY RULES (pick exactly one):
- "billing"  = anything about money: payment failed, invoices, charges, refunds, wrong amount, subscriptions, renewal
- "support"  = technical problems: bugs, crashes, login issues, API errors, data breach concerns, how-to questions, export questions
- "spam"     = unsolicited: ads, scams, get-rich-quick, pharmaceutical spam, fake prizes
- "inquiry"  = sales/business questions: pricing, plans, partnerships, feature requests, student discounts, newsletters

ROUTING RULES (must match category):
- billing category  -> "billing_team"
- support category  -> "support_team"  
- spam category     -> "trash"
- inquiry category  -> "sales_team"
  EXCEPTION: newsletters and resolved tickets -> "trash" even if category is inquiry/support

IMPORTANT EXCEPTIONS:
- "data breach" emails -> category="support", route_to="support_team", priority="urgent"
- "production down" or "API rate limit exceeded" -> priority="urgent", category="support"
- "enterprise 500 users" bulk inquiry -> priority="urgent", category="inquiry", route_to="sales_team"
- "newsletter" or "subscription renewal reminder" from the platform itself -> priority="low", category="inquiry", route_to="trash"
- "resolved ticket / closing ticket" -> priority="low", category="support", route_to="support_team"

Output ONLY this JSON. No markdown. No explanation. No backticks:
{{"priority": "...", "category": "...", "route_to": "..."}}

Email to classify:
FROM: {sender}
SUBJECT: {subject}
BODY: {body}"""


# ── LLM Call ──────────────────────────────────────────────

def call_llm(subject: str, body: str, sender: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(sender=sender, subject=subject, body=body)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=80,
        ),
    )
    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Task Runner ───────────────────────────────────────────

def run_task(task_id: str, verbose: bool = True) -> float:
    obs = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'='*60}")

    total_reward = 0.0
    step_count = 0

    while True:
        if obs.email_id == "DONE":
            break

        try:
            prediction = call_llm(obs.subject, obs.body, obs.sender)
        except Exception as e:
            if verbose:
                print(f"  [Step {step_count+1}] LLM error: {e}. Using fallback.")
            prediction = {"priority": "normal", "category": "support", "route_to": "support_team"}

        action = Action(
            priority=prediction.get("priority", "normal"),
            category=prediction.get("category", "support"),
            route_to=prediction.get("route_to", "support_team"),
        )

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        if verbose:
            all_correct = info["correct_priority"] and info["correct_category"] and info["correct_route"]
            status = "[OK]" if all_correct else "[--]"
            print(f"  {status} Email {step_count}: reward={reward:.2f} | "
                  f"priority={action.priority} (want: {info['label']['priority']}) | "
                  f"category={action.category} (want: {info['label']['category']}) | "
                  f"route={action.route_to} (want: {info['label']['route_to']})")

        if done:
            break

    final_score = round(total_reward / step_count, 4) if step_count > 0 else 0.0

    if verbose:
        print(f"\n  Final score: {final_score}")
        print(f"  Emails processed: {step_count}")

    return final_score


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Email Triage baseline agent")
    parser.add_argument("--task", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="Task to run (default: all)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set!")
        print("Get your FREE key at: https://aistudio.google.com/apikey")
        print("Then run: set GEMINI_API_KEY=AIza...")
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
        bar = "#" * int(score * 20)
        print(f"  {task_id:<8} {score:.4f}  {bar}")
    print()


if __name__ == "__main__":
    main()