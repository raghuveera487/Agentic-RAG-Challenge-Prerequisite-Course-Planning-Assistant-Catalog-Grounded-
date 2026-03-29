from dotenv import load_dotenv
load_dotenv()

import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class CoursePlanningAgent:
    def __init__(self):
        # ✅ Embeddings (FREE)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # ✅ Vector DB
        self.db = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=self.embeddings
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        # ✅ FREE LLM (NO OpenAI)
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Prompt
        self.prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context.

Context:
{context}

Question:
{question}

Instructions:
- Check prerequisites carefully
- If missing prerequisite → say NOT ELIGIBLE
- If satisfied → say ELIGIBLE
- If info missing → say NEED MORE INFO

Answer in this format:

Answer: <ELIGIBLE / NOT ELIGIBLE / NEED INFO>
Reason: <clear explanation>
""")

        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(
            f"{d.metadata.get('title')} ({d.metadata.get('url')}):\n{d.page_content}"
            for d in docs
        )

    def ask(self, query: str) -> str:
        return self.chain.invoke(query)


if __name__ == "__main__":
    print("🤖 Course Planning Assistant Initialized. Type 'quit' to exit.")

    agent = CoursePlanningAgent()

    while True:
        user_input = input("\nStudent: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        print("\n" + "="*50)
        response = agent.ask(user_input)
        print(response)
        print("="*50)
