'''
LockLM - LLM Backend
'''
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json

client = OpenAI()

# getting openai api key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4.1-mini"

# rubric negotation instructions for LockLM
RUBRIC_INSTRUCTIONS= """You are an evaluation plan generator. Given a users goal, create a rubric for assessing whether someone has made meaningful PROGRESS toward that goal. 

When the user gives you a goal, propose a rubric for checking their PROGRESS:
- Match the criteria to the goal. 
- Only include criteria that can be observed in a progress update. 
- Be specific to the goal, no generic filler criteria. 

Scoring scale for each criterion: 
1 = No meaningful attempt 
2 = Minimal effort, falls short of the gial 
3 = Adequate progress, genuine effort toward goal 
4 = Significant progress, goal is addressed
5 = Outstanding progress, goal is fully achieved or exceeded

After proposing, the user may want to adjust the rubric. They might say: 
- "Add a criterion about X"
- "Remove the criterion about Y"
- "That's too strict, make it more lenient."
- "Looks good, let's use this rubric."

Update the rubric based on the user's feedback, and when the user approves, finalize the rubric for use.
Output the final rubric between [FINAL RUBRIC] and [END RUBRIC] tags.

Only use these tags when the user has explicitly agreed."""


# phase for rubric negotation - user goes back and forth until the rubric is finalized. 
def negotiate_rubric(goal):
    conversation = [
        {"role": "user", "content": f"My goal is: {goal}\n\nPropose a rubric for checking my progress."}
    ]
    res = client.responses.create(
        model=MODEL,
        instructions=RUBRIC_INSTRUCTIONS,
        input=conversation,
        temperature=0.4,
    )

    reply = res.output_text.strip()

    conversation.append({"role": "assistant", "content": reply})
    print(f"\nLockLM: {reply}\n")

    while True: 
        user_input = input("You > ").strip()
        if not user_input:
            continue
 
        conversation.append({"role": "user", "content": user_input})
 
        res = client.responses.create(
            model=MODEL,
            instructions=RUBRIC_INSTRUCTIONS,
            input=conversation,
            temperature=0.4,
        )
 
        reply = res.output_text.strip()
        conversation.append({"role": "assistant", "content": reply})
        print(f"\nLockLM> {reply}\n")
 
        if "[FINAL RUBRIC]" in reply and "[END RUBRIC]" in reply:
            start = reply.index("[FINAL RUBRIC]")
            end = reply.index("[END RUBRIC]") + len("[END RUBRIC]")
            rubric = reply[start:end]
            print("Rubric finalized.\n")
            return rubric, conversation
 
 
 # coaching phase where LockLM gives feedback and guidance based on the rubric
def build_session_instructions(goal, rubric):
    return f"""You are LockLM, a progress accountability assistant.
 
THE USER'S GOAL:
"{goal}"
 
THE AGREED RUBRIC:
{rubric}
 
Your job:
- Help the user work toward their goal
- Answer questions, give tips, help them plan their approach
- You know what the rubric expects, so nudge them in the right direction
- Be concise and direct
- Do NOT do the work for them"""
 
 
def coach(message, conversation, instructions):
    conversation.append({"role": "user", "content": message})
 
    res = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=conversation,
        temperature=0.5,
    )
 
    reply = res.output_text.strip()
    conversation.append({"role": "assistant", "content": reply})
    return reply

 
GRADE_SCHEMA = {
    "type": "json_schema",
    "name": "evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "reasoning": {"type": "string"},
            "improvements": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["score", "reasoning", "improvements"],
        "additionalProperties": False
    }
}
 
 # grading phase to eveluate user's submission against the generated rubric
def grade(submission, conversation, instructions):
    grade_msg = f"""[GRADING MODE]
The user is submitting their work. Apply the agreed rubric to their submission.
 
Go through each part of the rubric and check whether the submission meets it.
Then give a final score (1-5) based on PROGRESS.
A rough but genuine attempt at the full goal deserves a 3.
 
Submission:
\"\"\"{submission}\"\"\""""
 
    conversation.append({"role": "user", "content": grade_msg})
 
    res = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=conversation,
        temperature=0.2,
        text={"format": GRADE_SCHEMA},
    )
 
    reply = res.output_text.strip()
    conversation.append({"role": "assistant", "content": reply})
 
    result = json.loads(reply)
    result["score"] = max(1, min(5, int(result.get("score", 1))))
    return result
 
 
def save_session(data):
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSession saved to {filename}")
 
 
 # running LockLM session
def main(): 
    goal = input("Enter your goal:\n> ").strip()
    if not goal:
        return
 
    print("\nLet's define your rubric together.")
    rubric, rubric_conversation = negotiate_rubric(goal)
 
    instructions = build_session_instructions(goal, rubric)
    conversation = []
 
    greeting = coach(
        "Rubric is locked. I'm ready to work.",
        conversation,
        instructions,
    )
    print(f"LockLM: {greeting}\n")
    print("Type normally to chat, 'submit' to submit, 'rubric' to review, 'quit' to exit\n")
 
    evaluations = []
    attempt = 0
 
    while True:
        user_input = input("You > ").strip()
        if not user_input:
            continue
 
        if user_input.lower() == "quit":
            break
 
        if user_input.lower() == "rubric":
            print(f"\n{rubric}\n")
            continue
 
        if user_input.lower() == "submit":
            print("Paste your work (type END on a new line when done):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            submission = "\n".join(lines)
 
            if not submission.strip():
                print("Empty submission.\n")
                continue
 
            attempt += 1
            print(f"\nGrading attempt {attempt}...\n")
 
            try:
                result = grade(submission, conversation, instructions)
            except Exception as e:
                print(f"Error: {e}\n")
                continue
 
            score = result["score"]
            evaluations.append({
                "attempt": attempt,
                "score": score,
                "reasoning": result["reasoning"],
                "improvements": result["improvements"],
            })
 
            print(f"Score: {score}/5")
            print(f"Reasoning: {result['reasoning']}")
            if result["improvements"]:
                print("Improvements:")
                for imp in result["improvements"]:
                    print(f"  - {imp}")
            print()
 
            if score >= 3:
                print("Progress verified!\n")
                agree = input("Do you agree with this evaluation? (y/n) > ").strip().lower()
                if agree == "y":
                    save_session({
                        "goal": goal,
                        "rubric": rubric,
                        "rubric_conversation": rubric_conversation,
                        "conversation": conversation,
                        "evaluations": evaluations,
                    })
                else:
                    print("Evaluation discarded.")
                break
            else:
                print("Below threshold.\n")
                reply = coach(
                    "I didn't pass. Based on the grading, what did I do well and what should I fix?",
                    conversation,
                    instructions,
                )
                print(f"LockLM: {reply}\n")
        else:
            reply = coach(user_input, conversation, instructions)
            print(f"LockLM: {reply}\n")
 
 
if __name__ == "__main__":
    main()
