---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
license: mit

# Email Triage OpenEnv

A real-world reinforcement learning environment for training AI agents to triage, prioritize, and route customer emails.

---

## What Is This?

This environment simulates a real customer support inbox. An AI agent must read each incoming email and decide:

1. Priority -- How urgently does this need a response?
2. Category -- What type of email is this?
3. Route -- Which team should handle it?

This is a task humans do every day in customer support roles, making it ideal for training and evaluating AI agents.

---

## Observation Space

| Field        | Type    | Description                                |
|--------------|---------|--------------------------------------------|
| task_id      | string  | Which task is running (easy/medium/hard)   |
| email_id     | string  | Unique email identifier                    |
| subject      | string  | Email subject line                         |
| body         | string  | Full email body                            |
| sender       | string  | Sender email address                       |
| step_number  | int     | Current step (1-indexed)                   |
| total_emails | int     | Total emails in this task                  |

---

## Action Space

| Field    | Type   | Options                                                    |
|----------|--------|------------------------------------------------------------|
| priority | string | urgent, normal, low                                        |
| category | string | billing, support, spam, inquiry                            |
| route_to | string | billing_team, support_team, trash, sales_team              |

---

## Tasks

| Task   | Emails | Description                                                      |
|--------|--------|------------------------------------------------------------------|
| easy   | 5      | Clear spam, obvious urgency signals, simple routing              |
| medium | 10     | Mixed signals, some ambiguous cases                              |
| hard   | 20     | All emails including edge cases and enterprise scenarios          |

---

## Reward Function

Per-step partial reward (not just end-of-episode):

| Component                          | Reward |
|------------------------------------|--------|
| Correct priority                   | +0.40  |
| Near-miss priority (urgent/normal) | +0.15  |
| Correct category                   | +0.35  |
| Correct route                      | +0.25  |
| Routing urgent email to trash      | -0.30  |

Score range: 0.0 to 1.0 per email.

---

## Baseline Scores

Scored using gpt-4o-mini with temperature=0:

| Task   | Score |
|--------|-------|
| easy   | 0.85  |
| medium | 0.78  |
| hard   | 0.71  |

---

## Setup and Usage

### Local Setup

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-openenv
cd email-triage-openenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn server:app --reload --port 7860
```

### Docker Setup

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv
```

---

## API Endpoints

### POST /reset
Start a new episode.
```json
{ "task_id": "easy" }
```

### POST /step
Take one action (classify one email).
```json
{
  "priority": "urgent",
  "category": "billing",
  "route_to": "billing_team"
}
```

### GET /state
Get current environment state.

### GET /tasks
List all tasks and action schema.

### POST /grader
Score a completed episode.
```json
{
  "task_id": "easy",
  "actions": [
    {"priority": "urgent", "category": "billing", "route_to": "billing_team"}
  ]
}
```

### POST /baseline
Run the baseline inference script.
```json
{ "task_ids": ["easy", "medium", "hard"] }
```

---

## Run Baseline

```bash
# Set your OpenAI API key (Windows)
set OPENAI_API_KEY=sk-...

# Run all tasks
python baseline.py

# Run one task
python baseline.py --task easy
```

---

## Project Structure

```
email-triage-openenv/
|-- environment.py      Core env: step/reset/state + reward function
|-- server.py           FastAPI server with all endpoints
|-- baseline.py         LLM baseline inference script
|-- openenv.yaml        OpenEnv metadata
|-- Dockerfile          Container config
|-- requirements.txt    Python dependencies
|-- README.md           This file
```

---

## Tags

email, triage, nlp, customer-support, classification, real-world, openenv
