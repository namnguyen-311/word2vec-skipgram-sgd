from transformers import pipeline

qa = pipeline("question-answering")

context = """
Word2Vec is a technique used to learn vector representations of words by predicting 
context words given a target word (Skip-Gram) or predicting the center word given the 
context (CBOW). It relies on dot products, softmax, and negative sampling to train.
"""

print("Ask me anything about Word2Vec! Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    result = qa(question=question, context=context)
    print("Bot:", result["answer"])
