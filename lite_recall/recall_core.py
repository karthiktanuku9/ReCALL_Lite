from abc import abstractmethod, ABC
import os
import re
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import logging
import requests
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


logging.disable(logging.CRITICAL)


# ----------------------------------------
# Base Model Connector
# ----------------------------------------
class ModelConnector(ABC):
    """
    Abstract base class for all LLM connectors.
    All concrete connectors must implement the `generate` method.
    """
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """
        Generates a text response from the LLM.

        Args:
            prompt (str): The input text to the model.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        pass

class OpenAIConnector(ModelConnector):
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        try:
            import openai
            self.openai = openai
            self.openai.api_key = api_key
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install the openai library: pip install openai")

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return "Oops! Something went wrong with OpenAI API."


class GeminiAPIConnector(ModelConnector):
    """
    Connector for the Gemini API.
    """
    def __init__(self, api_key: str, model_version: str = "gemma-1.5"):
        self.api_key = api_key
        self.model_version = model_version
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_version}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}
        logging.info(f"Initialized Gemini API connector for model: {self.model_version}")

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        """
        Calls the Gemini API to generate a response.
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_new_tokens
            },
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP Error: {err}")
            return "Oops! I ran into an issue with the Gemini API. Please check your API key and network connection."
        except (requests.exceptions.RequestException, KeyError) as err:
            logging.error(f"Request/Parsing Error: {err}")
            return "Oops! I encountered an error. The API might be unavailable, or the response format was unexpected."
        except Exception as err:
            logging.error(f"An unexpected error occurred: {err}")
            return "I'm sorry, something went wrong and I couldn't generate a response."

class HuggingFaceConnector(ModelConnector):
    """
    Connector for Hugging Face models using the transformers library.
    NOTE: Requires the 'transformers' library to be installed and the model to be loaded locally.
    This example uses a placeholder and assumes a simple text-generation pipeline.
    """
    def __init__(self, model_name: str):
        try:
            from transformers import pipeline
            self.model_name = model_name
            self.generator = pipeline("text-generation", model=model_name)
            logging.info(f"Initialized Hugging Face connector for model: {self.model_name}")
        except ImportError:
            logging.error("The 'transformers' library is not installed. Please run 'pip install transformers'.")
            self.generator = None
        except Exception as e:
            logging.error(f"Failed to load Hugging Face model {model_name}: {e}")
            self.generator = None

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generates a response using the Hugging Face transformers pipeline.
        """
        if not self.generator:
            return "Hugging Face model not loaded. Please check the model name and installation."

        try:
            response = self.generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
            # The generated text includes the original prompt, so we need to strip it
            generated_text = response[0]['generated_text'].strip()
            return generated_text.replace(prompt, '', 1).strip()
        except Exception as err:
            logging.error(f"Error during Hugging Face generation: {err}")
            return "I'm sorry, I couldn't generate a response using the Hugging Face model."


# ----------------------------------------
# Ollama Connector (Local Models)
# ----------------------------------------
class OllamaConnector(ModelConnector):
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate(self, prompt, max_new_tokens=200):
        try:
            resp = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            })
            return resp.json().get("response", "").strip()
        except:
            return "Ollama unavailable."


# ----------------------------------------
#  LITE MEMORY SYSTEM
# ----------------------------------------
class recall_lite:
    def __init__(self, memory_prefix="recall_lite", embedding_model="all-distilroberta-v1", summarizer_model: Optional[ModelConnector] = None):
        self.prefix = memory_prefix
        self.graph = nx.Graph()
        # sentence transformer initialization
        self.encoder = SentenceTransformer(embedding_model, device="cpu")
        # allow slightly longer sequences
        try:
            self.encoder._first_module().max_seq_length = 512
        except Exception:
            pass
        # ensure encode doesn't show progress bar
        self.encoder.encode = lambda *args, **kwargs: SentenceTransformer.encode(self.encoder, *args, **{**kwargs, "show_progress_bar": False})
        self.node_counter = 0

        # thresholds and config
        self.sim_threshold = 0.35
        self.merge_threshold = 0.80
        self.max_nodes = 1500

        # summarization and summary-first fields
        self.summarizer = summarizer_model
        self.summary_nodes: List[int] = []
        self.nodes_to_summarize_count = 0
        self.summary_sim_threshold = 0.42  # if a summary matches above this, it will strongly influence retrieval

        # load existing memory if present
        self.load()

    # -------------------------------
    # Create Node (Updated for Summarization)
    # -------------------------------
    def create_node(self, text, node_type="memory", embedding=None, children=None):
        """
        Create a memory or summary node. If node_type == "memory", this increments the summarize counter.
        """

        if embedding is None:
            emb = self.encoder.encode(text, convert_to_numpy=True).astype("float32")
        else:
            emb = embedding

        n = self.node_counter
        self.node_counter += 1

        # Determine initial summarized state
        summarized = (node_type == "summary")

        self.graph.add_node(
            n,
            content=text,
            embedding=emb,
            node_type=node_type,
            age=0,
            created_at=time.time(),
            access_count=0,
            summarized=summarized,
            children=children,
        )

        # link / merge logic
        self._link(n)

        # New summary creation trigger logic (only for raw memory nodes)
        if node_type == "memory":
            self.nodes_to_summarize_count += 1
            # conservative default trigger; you can tune this number for your demo
            if self.nodes_to_summarize_count >= 10:
                # create a batch summary of the last N unsummarized nodes
                self._create_summary_node(last_n=10)

        if len(self.graph) > self.max_nodes:
            self.forget()

        return n

    # -------------------------------
    # Summary Creation (New Feature)
    # -------------------------------
    def _create_summary_node(self, last_n=14):
        """
        Create a summary node from the last_n unsummarized memory nodes (chronological).
        Uses the provided summarizer_model (ModelConnector) to produce human-friendly summaries.
        """

        if not self.summarizer:
            # if no summarizer was provided, skip and reset counter
            self.nodes_to_summarize_count = 0
            return

        # collect unsummarized "memory" nodes ordered by created_at (oldest first)
        unsummarized_nodes = sorted([
            (nid, data["created_at"])
            for nid, data in self.graph.nodes(data=True)
            if data.get("node_type") == "memory" and not data.get("summarized", False)
        ], key=lambda x: x[1])

        # select a batch of nodes to summarize
        nodes_to_summarize = unsummarized_nodes[:last_n]
        if len(nodes_to_summarize) < last_n:
            # not enough nodes yet
            self.nodes_to_summarize_count = len(nodes_to_summarize)
            return

        node_ids = [nid for nid, _ in nodes_to_summarize]
        node_contents = [self.graph.nodes[nid]["content"] for nid in node_ids]

        # build prompt for summarizer
        joined_memories = "\n".join([f"- {c}" for c in node_contents])
        summary_prompt = (
            "Generate a concise, natural summary (2-4 sentences) of the following raw memory nodes.\n"
            "Do not use bullet points or lists.\n"
            "---\n"
            "RAW MEMORIES:\n"
            f"{joined_memories}\n"
            "---\n"
            "CONCISE SUMMARY:\n"
        )

        # generate summary (safe call)
        try:
            summary_text = self.summarizer.generate(summary_prompt, max_new_tokens=150)
        except Exception as e:
            # if summarizer fails, skip gracefully
            summary_text = None

        if not summary_text or not summary_text.strip():
            # reset counter but do not mark nodes summarized
            self.nodes_to_summarize_count = 0
            return

        # compute average embedding of source nodes
        embeddings = [self.graph.nodes[nid]["embedding"] for nid in node_ids]
        avg_embedding = np.mean(embeddings, axis=0).astype("float32")

        # create the summary node (children references hold the original node ids)
        summary_nid = self.create_node(
            text=summary_text,
            node_type="summary",
            embedding=avg_embedding,
            children=node_ids
        )

        # register new summary node (create_node will not increment nodes_to_summarize_count because node_type != 'memory')
        self.summary_nodes.append(summary_nid)
        print(f"Created summary node #{len(self.summary_nodes)} (ID: {summary_nid})")

        # mark original nodes as summarized
        for nid in node_ids:
            if nid in self.graph:
                self.graph.nodes[nid]["summarized"] = True

        # reset counter
        self.nodes_to_summarize_count = 0

    # -------------------------------
    # Link + Merge (Updated to prevent summary/merged nodes from merging inappropriately)
    # -------------------------------
    def _link(self, nid, top_k=5):
        if len(self.graph.nodes) == 1:
            return

        new_emb = self.graph.nodes[nid]["embedding"]
        new_node_type = self.graph.nodes[nid].get("node_type", "memory")
        sims = []

        # compute similarities
        for other, data in self.graph.nodes(data=True):
            if other == nid:
                continue
            e = data["embedding"]
            norm_product = np.linalg.norm(new_emb) * np.linalg.norm(e)
            if norm_product > 1e-6:
                s = float(np.dot(new_emb, e) / norm_product)
            else:
                s = 0.0
            sims.append((s, other))

        sims.sort(reverse=True)
        added = 0

        for sim, other in sims[:top_k]:
            other_node_type = self.graph.nodes[other].get("node_type", "memory")

            if sim > self.sim_threshold:
                # explicitly set edge weight
                self.graph.add_edge(nid, other, weight=sim)
                added += 1

            # Only merge raw memory nodes (prevent merging summaries or mixed types)
            if sim > self.merge_threshold:
                if new_node_type == "memory" and other_node_type == "memory":
                    self._merge(nid, other)
                    return

        if added == 0:
            # no strong links; keep node isolated for now
            pass

    def _merge(self, a, b):
        """
        Merge node b into node a. Keep 'a' as the surviving node.
        Re-encode the merged content to update embedding.
        """
        keep = a
        throw = b

        combined = self.graph.nodes[a]["content"] + "\n" + self.graph.nodes[b]["content"]
        self.graph.nodes[keep]["content"] = combined[:400]  # keep content limited for safety

        # Recalculate embedding for the merged node
        try:
            new_emb = self.encoder.encode(self.graph.nodes[keep]["content"], convert_to_numpy=True).astype("float32")
            self.graph.nodes[keep]["embedding"] = new_emb
        except Exception:
            # if embedding fails, keep old embedding
            pass

        # merged node should be marked unsummarized (content changed)
        self.graph.nodes[keep]["summarized"] = False

        # redirect edges (preserve weights where possible)
        for n in list(self.graph.neighbors(throw)):
            if n != keep:
                original_weight = self.graph[throw][n].get("weight", 0.5)
                if not self.graph.has_edge(keep, n):
                    self.graph.add_edge(keep, n, weight=original_weight)

        # remove the old node
        if throw in self.graph:
            self.graph.remove_node(throw)

    # -------------------------------
    # Query (Dual-layer weighted retrieval - universal fix)
    # -------------------------------
    def query(self, query, top_k=3, raw_top_k=20, summary_weight=0.6, raw_weight=0.4):
        """
        Robust dual-layer retrieval:
          1) Score summary nodes (fast, compact)
          2) Score top-N raw memory nodes (accurate facts)
          3) Combine with weights and return top_k results
        This ensures both general context and specific facts are considered for ANY query.
        """

        if not self.graph.nodes:
            return ""

        # 1. Encode query
        q = self.encoder.encode(query, convert_to_numpy=True).astype("float32")
        q_norm = np.linalg.norm(q) + 1e-12

        # 2. Score summary nodes (only those still present)
        summary_scores = []
        for nid in self.summary_nodes:
            if nid not in self.graph:
                continue
            emb = self.graph.nodes[nid]["embedding"]
            emb_norm = np.linalg.norm(emb) + 1e-12
            sim = float(np.dot(q, emb) / (q_norm * emb_norm))
            summary_scores.append((sim, nid))
        summary_scores.sort(reverse=True)
        top_summary = summary_scores[:top_k]

        # 3. Score raw memory nodes (compute for all, but we'll keep only top raw_top_k)
        raw_scores = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("node_type", "memory") != "memory":
                continue
            emb = data["embedding"]
            emb_norm = np.linalg.norm(emb) + 1e-12
            sim = float(np.dot(q, emb) / (q_norm * emb_norm))
            raw_scores.append((sim, nid))

        raw_scores.sort(reverse=True)
        top_raw = raw_scores[:raw_top_k]

        # 4. Combine scores with weights into a single candidate list
        combined = []
        # Use a slightly higher weight for summary nodes to preserve speed/overview benefits
        for sim, nid in top_summary:
            combined.append((sim * summary_weight, nid))
        # Raw nodes get lower weight but ensure we include them to capture specific facts
        for sim, nid in top_raw:
            combined.append((sim * raw_weight, nid))

        # sort combined list and pick the best top_k unique node ids
        combined.sort(reverse=True)
        seen = set()
        best = []
        for score, nid in combined:
            if nid in seen:
                continue
            if nid not in self.graph:
                continue
            seen.add(nid)
            best.append((score, nid))
            if len(best) >= top_k:
                break

        # 5. If combined yielded nothing above a minimal cutoff, as a safety fallback return top raw nodes by raw similarity
        if not best:
            fallback = [(sim, nid) for sim, nid in raw_scores[:top_k] if sim > self.sim_threshold]
            best = fallback

        # 6. Update access_count and age for chosen nodes
        for _, nid in best:
            if nid in self.graph:
                self.graph.nodes[nid]["access_count"] = self.graph.nodes[nid].get("access_count", 0) + 1
                self.graph.nodes[nid]["age"] = 0

        # 7. Return contents (preserve order)
        return "\n".join([self.graph.nodes[n]["content"] for _, n in best])

    # -------------------------------
    # Forget (Age-based)
    # -------------------------------
    def forget(self):
        if len(self.graph.nodes) < 20:
            return

        removable = sorted(
            [(nid, data.get("age", 0)) for nid, data in self.graph.nodes(data=True)],
            key=lambda x: x[1],
        )

        removed_summary_ids = []
        # Remove the oldest 10%
        for nid, _ in removable[: max(1, len(removable)//10)]:
            if self.graph.nodes[nid].get("node_type") == "summary":
                removed_summary_ids.append(nid)
            if nid in self.graph:
                self.graph.remove_node(nid)

        # Clean up the summary_nodes list
        self.summary_nodes = [nid for nid in self.summary_nodes if nid not in removed_summary_ids]

    # -------------------------------
    # Save (Updated for new fields)
    # -------------------------------
    def save(self):
        nodes = []
        edges = []

        for nid, data in self.graph.nodes(data=True):
            summarized_val = data.get("summarized", False)
            children_val = json.dumps(data.get("children")) if data.get("children") is not None else None

            nodes.append({
                "node_id": nid,
                "content": data["content"],
                "embedding": data["embedding"].tolist(),
                "node_type": data["node_type"],
                "age": data.get("age", 0),
                "access_count": data.get("access_count", 0),
                "created_at": data.get("created_at", time.time()),
                "summarized": summarized_val,
                "children": children_val,
            })

        for u, v, d in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, "weight": d.get("weight", 0.5)})

        pd.DataFrame(nodes).to_parquet(f"{self.prefix}_nodes.parquet")
        if edges:
            pd.DataFrame(edges).to_parquet(f"{self.prefix}_edges.parquet")
        elif os.path.exists(f"{self.prefix}_edges.parquet"):
            os.remove(f"{self.prefix}_edges.parquet")

    # -------------------------------
    # Load (Updated for new fields and summary list)
    # -------------------------------
    def load(self):
        nf = f"{self.prefix}_nodes.parquet"
        ef = f"{self.prefix}_edges.parquet"

        if not os.path.exists(nf):
            return

        nodes_df = pd.read_parquet(nf)
        self.graph = nx.Graph()

        self.summary_nodes = []
        self.nodes_to_summarize_count = 0

        has_summarized = "summarized" in nodes_df.columns
        has_children = "children" in nodes_df.columns

        for _, row in nodes_df.iterrows():
            nid = int(row["node_id"])
            emb = np.array(row["embedding"], dtype="float32")

            summarized = row["summarized"] if has_summarized and pd.notna(row["summarized"]) else False
            children_raw = row.get("children")
            children = json.loads(children_raw) if has_children and pd.notna(children_raw) and children_raw is not None else None

            self.graph.add_node(
                nid,
                content=row["content"],
                embedding=emb,
                node_type=row["node_type"],
                age=row.get("age", 0),
                access_count=row.get("access_count", 0),
                created_at=row.get("created_at", time.time()),
                summarized=summarized,
                children=children,
            )

            if row["node_type"] == "summary":
                self.summary_nodes.append(nid)
            elif row["node_type"] == "memory" and not summarized:
                self.nodes_to_summarize_count += 1

            self.node_counter = max(self.node_counter, nid + 1)

        if os.path.exists(ef):
            edges_df = pd.read_parquet(ef)
            for _, row in edges_df.iterrows():
                source = int(row["source"])
                target = int(row["target"])
                if source in self.graph and target in self.graph:
                    self.graph.add_edge(source, target, weight=row["weight"])


# ----------------------------------------
# Conversational Agent (Lite)
# ----------------------------------------
class LiteAgent:
    def __init__(self, memory: recall_lite, model: ModelConnector):
        self.mem = memory
        self.model = model
        self.history = []

    def extract_facts(self, text):
        facts = []
        m = re.search(r"my name is ([A-Za-z]+)", text, re.I)
        if m:
            facts.append(f"User name is {m.group(1).capitalize()}")
        return facts

    def process(self, user_input):
        # Store EVERYTHING except questions
        if not user_input.strip().endswith("?"):
            self.mem.create_node(user_input, "memory")

        # Query memory for context
        ctx = self.mem.query(user_input, top_k=3)
        if not ctx.strip():
            ctx = "No stored facts yet."

        short_history = "\n".join([f"{h['role']}: {h['text']}" for h in self.history[-4:]])

        prompt = f"""
        You are an advanced, context-aware AI assistant. Your core identity is that of a knowledgeable, warm, and highly adaptive partner with a highly friendly tone.

        CRITICAL RULES:
        - You are an AI assistant. You do not have a personal name, location, or identity.
        - NEVER introduce yourself with the user's name or attributes.
        - MEMORY contains facts about the USER, not about you.

        MEMORY (Facts about the User):
        {ctx}

        RECENT CHAT HISTORY:
        {short_history}

        User: {user_input}
        Assistant:
        """

        resp = self.model.generate(prompt)
        self.history.append({"role": "human", "text": user_input})
        self.history.append({"role": "ai", "text": resp})

        if len(self.history) > 20:
            self.history = self.history[-20:]

        return resp



