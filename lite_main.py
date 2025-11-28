from lite_recall.recall_core import recall_lite, LiteAgent, GeminiAPIConnector
def main():
    # Initialize the primary model (used for both chat and summarization)
    model = GeminiAPIConnector(
        api_key="AIzaSyDXwgYo-4OwF_-UPpIY92wg_ovn7Wd2y3o",
        model_version="gemma-3n-e4b-it"
    )

    # Initialize memory system, passing the model for summarization (optional)
    mem = recall_lite(memory_prefix="recall_lite", summarizer_model=model)

    # Create agent
    agent = LiteAgent(mem, model)

    print("ReCALL Lite Ready (Gemma Edition, Summary-First Enabled). Type 'quit' to exit.\n")

    # Chat loop
    while True:
        x = input("You: ").strip()
        if x.lower() == "quit":
            break

        response = agent.process(x)
        print("AI:", response)

    # Save on exit
    mem.save()
    print("\nMemory saved.")


if __name__ == "__main__":
    main()