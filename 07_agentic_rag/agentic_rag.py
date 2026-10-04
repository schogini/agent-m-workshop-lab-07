import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from agno.agent import Agent
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from common import EMBEDDING_MODEL, make_model

FAQ_PATH = ROOT / "05_rag" / "data" / "college_faq.md"

knowledge = Knowledge(
    vector_db=ChromaDb(
        collection="agent_m_agentic_rag",
        path=str(ROOT / "tmp" / "agentic_rag"),
        persistent_client=True,
        embedder=FastEmbedEmbedder(id=EMBEDDING_MODEL),
    )
)

knowledge.insert(path=str(FAQ_PATH), skip_if_exists=True)

agent = Agent(
    model=make_model(),
    knowledge=knowledge,
    search_knowledge=True,
    instructions=[
        "You are a Campus AI Buddy.",
        "Use knowledge search when a question depends on workshop facts.",
        "For general conversational questions, answer directly.",
        "Never invent workshop-specific facts.",
    ],
    markdown=True,
)

if __name__ == "__main__":
    questions = [
        "What is 7 times 8?",
        "Do I need Node.js for AGENT-M?",
        "What should participants bring?",
    ]

    for q in questions:
        print(f"\nQUESTION: {q}")
        agent.print_response(q, stream=True)
