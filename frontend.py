'''
LockLM - Gradio Frontend
'''


import json
import gradio as gr
import blocker


from llm_backend import (
    rubric_reply,
    extract_rubric,
    build_session_instructions,
    coach,
    grade,
    save_session,
)


def make_fresh_state():
    return {
        "phase": "goal",
        "goal": "",
        "rubric": "",
        "rubric_conversation": [],
        "session_instructions": "",
        "conversation": [],
        "evaluations": [],
        "attempt": 0,
        "blocked_apps": [],
        "blocker_running": False,
    }


def handle_goal(user_message, state, selected_apps):
    state["goal"] = user_message.strip()
    state["blocked_apps"] = selected_apps or []
    state["phase"] = "rubric"

    # Start blocker immediately when user enters goal
    if not state["blocker_running"] and state["blocked_apps"]:
        blocker.start(state["blocked_apps"])
        state["blocker_running"] = True

    state["rubric_conversation"] = [
        {"role": "user", "content": f"My goal is: {state['goal']}\n\nPropose a rubric for checking my progress."}
    ]

    reply = rubric_reply(state["rubric_conversation"])
    state["rubric_conversation"].append({"role": "assistant", "content": reply})

    return reply, state


def handle_rubric(user_message, state):

    state["rubric_conversation"].append({"role": "user", "content": user_message})

    reply = rubric_reply(state["rubric_conversation"])
    state["rubric_conversation"].append({"role": "assistant", "content": reply})

    rubric = extract_rubric(reply)
    if rubric:
        state["rubric"] = rubric
        state["session_instructions"] = build_session_instructions(state["goal"], rubric)
        state["phase"] = "working"

        greeting = coach(
            "Rubric is finalized. I'm ready to work with you!",
            state["conversation"],
            state["session_instructions"],
        )

        reply += f"\n\n---\n**Rubric finalized!**\n\n{greeting}"

    return reply, state


def handle_working(user_message, state):
    lower = user_message.strip().lower()

    if lower == "rubric":
        return f"**Your Rubric:**\n\n{state['rubric']}", state

    if lower == "submit":
        state["phase"] = "submitting"
        return "Paste your work below and send it.", state

    reply = coach(user_message, state["conversation"], state["session_instructions"])
    return reply, state


def handle_submit(user_message, state):
    try:
        result = grade(user_message.strip(), state["conversation"], state["session_instructions"])
    except Exception as e:
        state["phase"] = "working"
        return f"Error: {e}\n\nType **submit** to try again.", state

    state["attempt"] += 1
    state["evaluations"].append({
        "attempt": state["attempt"],
        "score": result["score"],
        "reasoning": result["reasoning"],
        "improvements": result["improvements"],
    })

    score = result["score"]
    improvements = "\n".join(f"- {imp}" for imp in result["improvements"])

    reply = f"**Score: {score}/5** (Attempt {state['attempt']})\n\n"
    reply += f"**Reasoning:** {result['reasoning']}\n\n"
    reply += f"**Improvements:**\n{improvements}\n\n"

    if score >= 3:
        reply += "**Progress verified!**\n\nDo you agree with this evaluation? Type **yes** to save or **no** to keep working."
        state["phase"] = "confirm"
    else:
        reply += "**Below threshold.**\n\nDo you agree with this evaluation? Type **yes** or **no**."
        state["phase"] = "confirm_fail"

    return reply, state


def handle_confirm(user_message, state):
    if user_message.strip().lower() in ("yes", "y"):
        filename = save_session({
            "goal": state["goal"],
            "rubric": state["rubric"],
            "rubric_conversation": state["rubric_conversation"],
            "conversation": state["conversation"],
            "evaluations": state["evaluations"],
        })
        state["phase"] = "done"
        blocker.stop()
        return f"Great work! Thank you for using LockLM!", state
    else:
        state["phase"] = "working"
        return "Evaluation discarded. Keep chatting or type **submit** to resubmit.", state


def handle_confirm_fail(user_message, state):
    if user_message.strip().lower() in ("yes", "y"):
        note = "Score noted."
    else:
        note = "Evaluation discarded."

    reply = coach(
        "I didn't pass. Based on the grading, what did I do well and what should I fix?",
        state["conversation"],
        state["session_instructions"],
    )
    state["phase"] = "working"
    return f"{note}\n\n{reply}\n\nType **submit** when ready to try again.", state


HANDLERS = {
    "goal": handle_goal,
    "rubric": handle_rubric,
    "working": handle_working,
    "submitting": handle_submit,
    "confirm": handle_confirm,
    "confirm_fail": handle_confirm_fail,
}

PHASE_INFO = {
    "goal": ("Instructions: Enter your goal", "Type your goal here..."),
    "rubric": ("Instructions: Negotiate your rubric. Tell LockLM when you are happy with it!", "Adjust the rubric or approve it..."),
    "working": ("Instructions: Work on your task. Type 'submit' when ready, or ask for help", "Ask for help or type 'submit'..."),
    "submitting": ("Instructions: Paste your work below", "Paste your work here..."),
    "confirm": ("Instructions: Do you agree with the evaluation?", "Type yes or no..."),
    "confirm_fail": ("Instructions: Do you agree this score is fair?", "Type yes or no..."),
    "done": ("Session complete", ""),
}


def respond(user_message, chat_history, state, selected_apps):
    if not user_message.strip():
        return "", chat_history, state

    if state["phase"] == "done":
        chat_history.append((user_message, "Session complete. Refresh to start a new one."))
        return "", chat_history, state

    handler = HANDLERS.get(state["phase"])
    current_phase = state["phase"]
    if handler:
        if current_phase == "goal":
            reply, state = handler(user_message, state, selected_apps)
        else:
            reply, state = handler(user_message, state)

        if current_phase == "submitting":
            display = f"[Submission Attempt {state['attempt']} Received]"
        else:
            display = user_message

        chat_history.append((display, reply))

    return "", chat_history, state


def build_ui():
    with gr.Blocks(
        title="LockLM",
        theme=gr.Theme.from_hub("hmb/midnight"),
    ) as app:

        state = gr.State(make_fresh_state())

        app_list = gr.CheckboxGroup(
        choices=["Discord", "Steam", "Chrome", "Spotify"],
        label="Select apps to block during focus mode"
        )

        gr.HTML("""
        <div class="header">
            <h1>🤖 LockLM</h1>
            <p>LLM-as-a-Judge for Task Accountability</p>
        </div>
        """)

        phase_display = gr.Markdown(
            value="<div class='phase-bar'>Instructions: Enter your goal</div>"
        )

        chatbot = gr.Chatbot(height=500, show_label=False, bubble_full_width=False)

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your goal here...",
                show_label=False, scale=9, container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        def on_send(user_message, chat_history, state, selected_apps):
            _, updated_history, updated_state = respond(user_message, chat_history, state, selected_apps)
            label, placeholder = PHASE_INFO.get(updated_state["phase"], ("", ""))
            phase_md = f"<div class='phase-bar'>{label}</div>"
            return "", updated_history, updated_state, phase_md, gr.update(placeholder=placeholder)

        send_btn.click(
            fn=on_send,
            inputs=[msg_input, chatbot, state, app_list],
            outputs=[msg_input, chatbot, state, phase_display, msg_input],
        )

        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, chatbot, state, app_list],
            outputs=[msg_input, chatbot, state, phase_display, msg_input],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch()
