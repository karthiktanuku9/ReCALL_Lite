
# **ReCALL Lite — A Lightweight Memory Engine for LLMs**

*A simple, powerful, human-like memory layer for everyday LLM applications.*

---

## 🚀 **What Is ReCALL?**

ReCALL is a **human-inspired memory mechanism** created by **Project Genesis** to address one of the biggest limitations of modern language models:

👉 **LLMs forget everything once the context window ends.**

ReCALL introduces a persistent, structured, cortex-like memory layer that allows AI systems to:

* store information across sessions
* retrieve memories based on meaning
* summarize and compress older memories
* merge similar information
* maintain stable long-term knowledge

It acts like a **mini cognitive memory organ**, letting LLMs recall facts the way humans do — not by scrolling back in history, but by semantic association.

ReCALL is one of Project Genesis’s core innnovations.

---

## ⚡ **What Is ReCALL Lite?**

ReCALL Lite is the **public, simplified version** of the original ReCALL memory system — fully rebuilt to:

* be extremely lightweight
* run on CPU
* work with any LLM (OpenAI, Gemini, HuggingFace, Ollama)
* require no complex setup
* give developers plug-and-play memory for their models

ReCALL Lite includes:

* **long-term structured memory nodes**
* **semantic retrieval**
* **automatic summarization**
* **graph-based linking of related memory**
* **node merging to reduce redundancy**
* **multi-session persistence (save/load)**

It’s simple enough for anyone to use, but powerful enough to transform how LLMs handle memory in real apps.

---

## 📌 **Benchmark: ReCALL Lite vs Normal Context Window**

Using **100 contexts** and **100 questions** on **Gemma 3B**, with 10-second intervals:

| System                    | Accuracy |
| ------------------------- | -------- |
| **Normal Context Window** | ~20%     |
| **ReCALL Lite**           | ~79%     |

This shows how even a lightweight memory layer can significantly boost factual retention.

> **Note:**
> Due to limited compute resources, the Project Genesis team was not able to run large-scale benchmarks at this stage.
> This is an **initial testimonial benchmark**, and full standardized benchmarking will begin soon as Genesis scaling continues.

---

## 🎯 **Why Project Genesis Built ReCALL Lite**

**Project Genesis** created ReCALL Lite for one important mission:

### 👉 **To open the door for researchers, engineers, and AI enthusiasts to experience what an AI-generated memory framework looks like — and build on it.**

ReCALL Lite serves as:

* a **teaching tool**
* a **research starting point**
* a **minimal reproduction** of a much larger memory architecture
* a **collaboration bridge** between the open-source community and Project Genesis

ReCALL (the original) is a major AGI subsystem.
ReCALL Lite allows the world to:

* experiment with the concepts
* understand the design philosophy
* explore long-term memory for LLMs
* contribute to future development

It’s not the full system — but it’s the **first accessible step** into cognitive memory engineering.

---

## 🧩 **How ReCALL Lite Helps Everyday LLM Developers**

Modern LLMs:

* lose older messages
* can’t persist information across sessions
* rely only on the current context window

ReCALL Lite upgrades any LLM into a **memory-capable agent** that can:

* remember user info
* recall facts days later
* summarize conversations into memory chunks
* retrieve relevant knowledge automatically
* persist knowledge between runs

### Perfect for:

* AI companions
* chatbots
* productivity assistants
* long-term agents
* research tools
* custom RAG alternatives
* autonomous workflows

If you’ve ever wanted your model to “remember stuff like a human,”
ReCALL Lite is built exactly for that.

---

# 👨‍💻 **How Developers Can Use ReCALL Lite**

A simple example:

```python
from recall_lite import ReCALLLite, LiteAgent, GeminiAPIConnector

model = GeminiAPIConnector(api_key="YOUR_KEY", model_version="gemma-3n-e4b-it")
memory = ReCALLLite(memory_prefix="recall_lite", summarizer_model=model)
agent = LiteAgent(memory, model)

response = agent.process("My name is Krishna.")
print(response)

```

ReCALL Lite handles:

* memory storage
* summaries
* retrieval
* merging
* forgetting
* context injection

Automatically.

---

## 🤝 **ReCALL Lite & Project Genesis**

ReCALL Lite is compatible with — and inspired by — the original ReCALL system built by Project Genesis.

### Project Genesis aims to solve AGI’s memory pillar.

ReCALL is part of that pursuit, designed to bring:

* long-term retention
* structured memory
* evolving knowledge
* cortex-like organization
* multi-session continuity

ReCALL Lite is the **open-source bridge** toward that vision — giving the world a glimpse into the architectures Genesis is creating.

---

## 🌐 **Why the Full ReCALL System Is Far More Powerful**

ReCALL Lite is only the *accessible public layer* — a simplified version designed for everyday developers.
The **original ReCALL**, created inside Project Genesis, is an advanced AGI-inspired memory architecture that goes far beyond what this lightweight version provides.

While Lite handles basic long-term storage, summarization, and semantic retrieval…

### 🔥 The full ReCALL system implements a much deeper cognitive model:

* **multi-layered memory organization inspired by human cortex structures**
* **adaptive retention and forgetting tuned for long-term stability**
* **dynamic reinforcement signals that strengthen important memories**
* **protection layers for critical knowledge**
* **context-aware memory routing**
* **hierarchical linking that evolves with use**

These capabilities position ReCALL as one of the first practical frameworks aimed at solving **the memory pillar of AGI**.

### ⭐ Why this matters

AGI needs four foundations:

1. **Memory**
2. **Reasoning**
3. **Perception**
3. **Consciousness**

ReCALL directly targets the first pillar — enabling an AI system to develop the kind of long-term, structured, continuously-evolving memory humans rely on.

It’s not just storage.
It’s not just RAG.
It’s **cognitive memory engineering**.

---

## 🤝 **Open Call for Collaborations & Partnerships**

ReCALL Lite is intentionally open-sourced so universities, engineers, startups, and independent researchers can:

* understand the direction of cognitive memory frameworks
* experiment with a working minimal version
* contribute ideas and enhancements
* propose research partnerships
* help shape the next generation of memory systems

The **full ReCALL system**, however, remains internal to Project Genesis and is reserved for:

* strategic partnerships
* research collaborations
* AGI labs
* institutions working on cognitive architectures
* organizations building autonomous agents

ReCALL Lite is the **gateway**.
By working with it, developers can demonstrate interest, provide feedback, and potentially become part of the teams testing or extending the full ReCALL framework.

---

## 🌌 **What Is Project Genesis?**

Project Genesis is the world’s first autonomous AI research engine designed to:

* **invent new frameworks**
* **build working systems**
* **evaluate its own designs**
* **evolve them automatically**
* **explore AI Frameworks, quantum models, and drug formulation discoveries**

Genesis autonomously generated:

* ARF-OP
* MSGL
* CasualLite
* ReCALL
* NoToxic
* CogniMesh
* EvoSoul
* AIDRA

And more.

Project Genesis is not a chatbot.
It is an **autonomous scientific partner** — built to explore the foundations of Scientific and Technical Discoveries.

ReCALL Lite is your invitation to join this mission.

---
