"""
Email Triage & Prioritization OpenEnv Environment
==================================================
A real-world RL environment where an AI agent learns to:
  - Read emails
  - Assign priority (urgent / normal / low)
  - Assign category (billing / support / spam / inquiry)
  - Route to the correct department

step() / reset() / state() follow the OpenEnv spec.
"""

import random
from typing import Any
from pydantic import BaseModel

# ─────────────────────────────────────────────
# 1. Typed Models  (OpenEnv spec requires these)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    email_id: str
    subject: str
    body: str
    sender: str
    step_number: int
    total_emails: int

class Action(BaseModel):
    priority: str       # "urgent" | "normal" | "low"
    category: str       # "billing" | "support" | "spam" | "inquiry"
    route_to: str       # "billing_team" | "support_team" | "trash" | "sales_team"

class RewardInfo(BaseModel):
    score: float
    done: bool
    correct_priority: bool
    correct_category: bool
    correct_route: bool
    feedback: str


# ─────────────────────────────────────────────
# 2. Email Dataset  (realistic fake emails)
# ─────────────────────────────────────────────

EMAIL_DATASET = [
    {
        "id": "e001",
        "subject": "URGENT: Payment failed - Account will be suspended",
        "body": "My payment failed 3 times and my account is about to be suspended. Please fix this immediately or I will cancel.",
        "sender": "angry.customer@gmail.com",
        "label_priority": "urgent",
        "label_category": "billing",
        "label_route": "billing_team",
    },
    {
        "id": "e002",
        "subject": "Question about your pricing plans",
        "body": "Hi, I'm interested in upgrading my plan. Could you send me information about the enterprise tier?",
        "sender": "prospect@company.com",
        "label_priority": "normal",
        "label_category": "inquiry",
        "label_route": "sales_team",
    },
    {
        "id": "e003",
        "subject": "Congratulations! You've won $1,000,000!!!",
        "body": "Click here to claim your prize. Limited time offer. Send your bank details now.",
        "sender": "winner@totallylegit.biz",
        "label_priority": "low",
        "label_category": "spam",
        "label_route": "trash",
    },
    {
        "id": "e004",
        "subject": "App crashes when I upload files",
        "body": "Every time I try to upload a PDF the app crashes. This has been happening since yesterday. I'm on Windows 11.",
        "sender": "user123@outlook.com",
        "label_priority": "urgent",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e005",
        "subject": "Invoice #4521 attached",
        "body": "Please find attached the invoice for last month's usage. Payment due in 30 days.",
        "sender": "billing@vendor.com",
        "label_priority": "normal",
        "label_category": "billing",
        "label_route": "billing_team",
    },
    {
        "id": "e006",
        "subject": "How do I export my data?",
        "body": "I've been looking through the docs but can't figure out how to export my data to CSV. Can you help?",
        "sender": "helpme@yahoo.com",
        "label_priority": "normal",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e007",
        "subject": "REFUND REQUEST - charged twice",
        "body": "I was charged twice for my subscription this month. I need an immediate refund. Order ID: 78234.",
        "sender": "dupcharge@gmail.com",
        "label_priority": "urgent",
        "label_category": "billing",
        "label_route": "billing_team",
    },
    {
        "id": "e008",
        "subject": "Free V1agra and more!!!",
        "body": "Best deals on pharmaceutical products. No prescription needed. Click now.",
        "sender": "deals@spam99.ru",
        "label_priority": "low",
        "label_category": "spam",
        "label_route": "trash",
    },
    {
        "id": "e009",
        "subject": "Partnership opportunity",
        "body": "Hi, I represent a mid-size company and we are exploring integration partnerships. Would love to schedule a call.",
        "sender": "partnerships@bigcorp.com",
        "label_priority": "normal",
        "label_category": "inquiry",
        "label_route": "sales_team",
    },
    {
        "id": "e010",
        "subject": "Login not working - locked out of account",
        "body": "I cannot log in at all. Tried resetting password 5 times. Still getting error 403. I have a deadline today!",
        "sender": "lockedout@work.com",
        "label_priority": "urgent",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e011",
        "subject": "Newsletter: Tips for using our platform",
        "body": "Here are 5 tips to get the most out of your subscription this month. Read our blog for more updates.",
        "sender": "newsletter@ourplatform.com",
        "label_priority": "low",
        "label_category": "inquiry",
        "label_route": "trash",
    },
    {
        "id": "e012",
        "subject": "Data breach concern - my info may be leaked",
        "body": "I saw news about a data breach and I'm worried my personal data might be compromised. Please advise urgently.",
        "sender": "worried@privacy.net",
        "label_priority": "urgent",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e013",
        "subject": "Can I get a student discount?",
        "body": "I'm a university student and was wondering if you offer any student discounts on your plans.",
        "sender": "student@university.edu",
        "label_priority": "low",
        "label_category": "inquiry",
        "label_route": "sales_team",
    },
    {
        "id": "e014",
        "subject": "Wrong amount on my bill",
        "body": "My bill shows $299 but I'm on the $99 plan. Please correct this before you charge my card on Friday.",
        "sender": "wrongbill@hotmail.com",
        "label_priority": "urgent",
        "label_category": "billing",
        "label_route": "billing_team",
    },
    {
        "id": "e015",
        "subject": "Feature request: dark mode",
        "body": "I'd love to see dark mode added to the app. Would really help reduce eye strain during late night sessions.",
        "sender": "darkmodefan@gmail.com",
        "label_priority": "low",
        "label_category": "inquiry",
        "label_route": "sales_team",
    },
    {
        "id": "e016",
        "subject": "API rate limit exceeded - production down",
        "body": "Our production system is hitting your API rate limits and is completely down. We're losing money every minute. URGENT.",
        "sender": "devops@startup.io",
        "label_priority": "urgent",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e017",
        "subject": "Re: Your recent support ticket",
        "body": "Thank you for getting back to me. The issue is now resolved. You can close the ticket.",
        "sender": "happycustomer@gmail.com",
        "label_priority": "low",
        "label_category": "support",
        "label_route": "support_team",
    },
    {
        "id": "e018",
        "subject": "Bulk license inquiry for 500 users",
        "body": "We are evaluating your platform for enterprise deployment across 500 users. Please send pricing and SLA details.",
        "sender": "procurement@enterprise.com",
        "label_priority": "urgent",
        "label_category": "inquiry",
        "label_route": "sales_team",
    },
    {
        "id": "e019",
        "subject": "MAKE MONEY FAST WORKING FROM HOME",
        "body": "Earn $5000 per week from home. No experience needed. Join thousands who are already making money!",
        "sender": "money@getrichfast.net",
        "label_priority": "low",
        "label_category": "spam",
        "label_route": "trash",
    },
    {
        "id": "e020",
        "subject": "Subscription renewal reminder",
        "body": "Your annual subscription renews in 7 days. No action needed if you'd like to continue.",
        "sender": "noreply@ourplatform.com",
        "label_priority": "normal",
        "label_category": "billing",
        "label_route": "billing_team",
    },
]


# ─────────────────────────────────────────────
# 3. The Environment Class
# ─────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    OpenEnv-compliant Email Triage Environment.
    The agent processes one email per step.
    Episode ends when all emails in the task batch are processed.
    """

    TASK_CONFIG = {
        "easy": {
            "description": "Triage 5 simple emails. Clear spam, obvious urgency.",
            "email_ids": ["e001", "e003", "e004", "e005", "e019"],
            "difficulty": "easy",
        },
        "medium": {
            "description": "Triage 10 emails with mixed signals and ambiguous cases.",
            "email_ids": ["e001", "e002", "e003", "e006", "e007", "e009", "e010", "e011", "e013", "e017"],
            "difficulty": "medium",
        },
        "hard": {
            "description": "Triage 20 emails including edge cases, nuanced priorities, and enterprise scenarios.",
            "email_ids": [e["id"] for e in EMAIL_DATASET],  # all 20
            "difficulty": "hard",
        },
    }

    def __init__(self):
        self._task_id = "easy"
        self._emails = []
        self._current_index = 0
        self._scores = []
        self._done = False
        self._email_lookup = {e["id"]: e for e in EMAIL_DATASET}

    # ── OpenEnv required methods ──

    def reset(self, task_id: str = "easy") -> Observation:
        """Start a new episode for the given task."""
        if task_id not in self.TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_id}. Choose from {list(self.TASK_CONFIG.keys())}")

        self._task_id = task_id
        config = self.TASK_CONFIG[task_id]
        self._emails = [self._email_lookup[eid] for eid in config["email_ids"]]
        self._current_index = 0
        self._scores = []
        self._done = False

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Process one email with the agent's action."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        email = self._emails[self._current_index]
        reward, info = self._compute_reward(action, email)
        self._scores.append(reward)

        self._current_index += 1
        self._done = self._current_index >= len(self._emails)

        if self._done:
            obs = self._make_observation(final=True)
        else:
            obs = self._make_observation()

        return obs, reward, self._done, info

    def state(self) -> dict:
        """Return current environment state (OpenEnv spec)."""
        return {
            "task_id": self._task_id,
            "current_email_index": self._current_index,
            "total_emails": len(self._emails),
            "scores_so_far": self._scores,
            "average_score": round(sum(self._scores) / len(self._scores), 4) if self._scores else 0.0,
            "done": self._done,
        }

    # ── Task & Grader methods ──

    def get_tasks(self) -> list[dict]:
        """Return all tasks with their action schema."""
        return [
            {
                "task_id": tid,
                "description": cfg["description"],
                "difficulty": cfg["difficulty"],
                "num_emails": len(cfg["email_ids"]),
                "action_schema": {
                    "priority": ["urgent", "normal", "low"],
                    "category": ["billing", "support", "spam", "inquiry"],
                    "route_to": ["billing_team", "support_team", "trash", "sales_team"],
                },
            }
            for tid, cfg in self.TASK_CONFIG.items()
        ]

    def grade_episode(self, task_id: str, actions: list[dict]) -> float:
        """
        Programmatic grader: replays a list of actions and returns a score 0.0–1.0.
        Used by the /grader endpoint.
        """
        obs = self.reset(task_id)
        total_score = 0.0

        for action_dict in actions:
            action = Action(**action_dict)
            obs, reward, done, info = self.step(action)
            total_score += reward
            if done:
                break

        num_emails = len(self.TASK_CONFIG[task_id]["email_ids"])
        return round(total_score / num_emails, 4)

    # ── Private helpers ──

    def _make_observation(self, final: bool = False) -> Observation:
        if final or self._current_index >= len(self._emails):
            # Return a terminal observation
            return Observation(
                task_id=self._task_id,
                email_id="DONE",
                subject="Episode Complete",
                body=f"All emails processed. Average score: {self.state()['average_score']}",
                sender="system",
                step_number=self._current_index,
                total_emails=len(self._emails),
            )

        email = self._emails[self._current_index]
        return Observation(
            task_id=self._task_id,
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            step_number=self._current_index + 1,
            total_emails=len(self._emails),
        )

    def _compute_reward(self, action: Action, email: dict) -> tuple[float, dict]:
        """
        Partial reward function — signals throughout the trajectory.
        Priority: 0.4 pts | Category: 0.35 pts | Route: 0.25 pts
        """
        reward = 0.0
        correct_priority = action.priority == email["label_priority"]
        correct_category = action.category == email["label_category"]
        correct_route = action.route_to == email["label_route"]

        if correct_priority:
            reward += 0.40
        elif self._is_close_priority(action.priority, email["label_priority"]):
            reward += 0.15  # partial credit for near-miss

        if correct_category:
            reward += 0.35

        if correct_route:
            reward += 0.25

        # Penalty: routing urgent email to trash
        if email["label_priority"] == "urgent" and action.route_to == "trash":
            reward -= 0.30

        # Clamp to [0.0, 1.0]
        reward = round(max(0.0, min(1.0, reward)), 4)

        info = {
            "email_id": email["id"],
            "correct_priority": correct_priority,
            "correct_category": correct_category,
            "correct_route": correct_route,
            "label": {
                "priority": email["label_priority"],
                "category": email["label_category"],
                "route_to": email["label_route"],
            },
            "reward": reward,
        }
        return reward, info

    def _is_close_priority(self, predicted: str, actual: str) -> bool:
        """Normal vs urgent is closer than low vs urgent."""
        close_pairs = {("urgent", "normal"), ("normal", "urgent"), ("normal", "low"), ("low", "normal")}
        return (predicted, actual) in close_pairs
