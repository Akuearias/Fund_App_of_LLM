from openai import OpenAI
from pinecone_memory import build_memory_store

client = OpenAI()
memory = build_memory_store()


def build_prompt(goal, submission, few_shot):
    return f"""
You are a strict evaluator.

Evaluate based on:
- Relevance
- Completeness
- Depth
- Clarity

A score of 3 means minimum acceptable progress.
Be strict.

Here are past examples:
{few_shot}

Now evaluate:

Goal:
{goal}

Submission:
{submission}

Output:
- score (1-5)
- feedback
"""


def evaluate(goal, submission):
    # 🔥 用 memory 取 few-shot
    matches = memory.retrieve_similar(goal, submission)

    few_shot = memory.build_few_shot_block(matches)

    prompt = build_prompt(goal, submission, few_shot)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content