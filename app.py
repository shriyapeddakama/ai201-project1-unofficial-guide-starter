"""Milestone 5: Gradio web interface for the CMU Unofficial Guide.

A viewer enters a question; the app retrieves relevant chunks, generates a
grounded answer with Groq, and shows both the answer and the source documents
it drew from.

Run from the project root (needs GROQ_API_KEY in .env):
    python app.py
Then open http://localhost:7860
"""

import gradio as gr

from src.generate import ask

EXAMPLES = [
    "What do students say are good study spots around campus?",
    "What can students do for free with their Andrew ID?",
    "What housing options do first year graduate students have?",
    "Which bus lines go from campus toward Shadyside?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the guide didn't have enough information)"
    return result["answer"], sources


with gr.Blocks(title="CMU Unofficial Guide") as demo:
    gr.Markdown(
        "# 🏫 The CMU Unofficial Guide\n"
        "Ask about CMU campus life — housing, dining, study spots, free perks, "
        "getting around Pittsburgh. Answers come **only** from collected student "
        "guides and Reddit threads, with sources shown."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What are good study spots on campus?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
