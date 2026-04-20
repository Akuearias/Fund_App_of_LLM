# seed_examples.py
from pinecone_memory import build_memory_store, MemoryExample

store = build_memory_store()

examples = [
    # ======================
    # High score: 5
    # ======================
    MemoryExample(
        example_id="seed-high-1",
        goal="Write a 500 word essay on Large Language Models",
        submission=(
            "Large Language Models (LLMs) are neural networks trained on massive amounts of text data. "
            "They typically use transformer architectures and learn to predict the next token in a sequence. "
            "This training objective allows them to generate coherent text, answer questions, summarize documents, "
            "and assist with coding tasks. A major strength of LLMs is their flexibility across many tasks without "
            "task-specific retraining. However, they can also produce inaccurate or biased outputs, which means users "
            "must evaluate them carefully. Overall, LLMs are powerful systems with broad real-world applications."
        ),
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=5,
        feedback="Fully addresses the topic with clear structure, specific technical details, and solid coverage of the main ideas.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),

    # High score: 4
    MemoryExample(
        example_id="seed-high-2",
        goal="Write a 500 word essay on Large Language Models",
        submission=(
            "Large language models are AI systems that generate text based on patterns learned from large datasets. "
            "They are used in chatbots, translation, summarization, and search. Their main advantage is that they can "
            "handle many language tasks using a single model. Still, they may hallucinate, reflect bias, or struggle "
            "with factual accuracy. These weaknesses show why human oversight remains important. In summary, LLMs are "
            "useful and influential, but they also require careful deployment."
        ),
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=4,
        feedback="Relevant and complete enough to justify progress, though the analysis is somewhat high-level and could go deeper.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),

    # ======================
    # Middle score: 3
    # ======================
    MemoryExample(
        example_id="seed-mid-1",
        goal="Write a 500 word essay on Large Language Models",
        submission=(
            "Large language models are AI tools that can generate text and answer questions. "
            "They are important because many people use them in daily work and study. "
            "They can help with writing, summarizing, and finding information."
        ),
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=3,
        feedback="Relevant and on-topic, but too shallow to be strong. It shows basic progress, not a polished final essay.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),

    MemoryExample(
        example_id="seed-mid-2",
        goal="Write a 500 word essay on Large Language Models",
        submission=(
            "Large Language Models are trained on large datasets and are capable of producing human-like text. "
            "They are used in many products today. One challenge is that they may sometimes produce incorrect answers."
        ),
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=3,
        feedback="The submission is correct and relevant, but it is underdeveloped. It reaches the minimum acceptable threshold only.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),

    # ======================
    # Low score: 2
    # ======================
    MemoryExample(
        example_id="seed-low-1",
        goal="Write a 500 word essay on Large Language Models",
        submission="LLMs are AI models that generate text. They are useful.",
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=2,
        feedback="Too brief to count as meaningful progress. It is related to the topic, but does not provide enough content to unlock apps.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),

    # Low score: 1
    MemoryExample(
        example_id="seed-low-2",
        goal="Write a 500 word essay on Large Language Models",
        submission="Technology is changing society in many ways, and people use computers every day.",
        rubric="Relevance, completeness, depth, clarity. Be strict. Score 3 only if the submission shows real progress.",
        score=1,
        feedback="Off-topic and does not address Large Language Models. This does not demonstrate progress toward the assigned task.",
        accepted=True,
        task_type="essay",
        rubric_version="v1",
    ),
]

store.store_examples(examples)
print(f"Seeded {len(examples)} examples into Pinecone.")