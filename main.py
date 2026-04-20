from judge import evaluate

goal = "Write a 500 word essay on Large Language Models"

submission = input("Enter your submission:\n")

result = evaluate(goal, submission)

print(result)

# Followed by:
# if score >= 3: unlock()
# else: retry()