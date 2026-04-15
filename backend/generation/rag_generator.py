"""
RAG Generator
==============
Generates answers using an LLM with retrieved context passages.
Supports OpenAI, Ollama, and HuggingFace backends.
"""

from typing import Optional
import json
import nltk
from loguru import logger
import math

# Download punkt tokenizer on first run if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, HF_MODEL_NAME,
)
from generation.prompt_templates import get_system_prompt, get_qa_prompt


class RAGGenerator:
    """
    LLM-based answer generator with strict grounding.
    Supports multiple backends: OpenAI, Ollama, HuggingFace.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = "huggingface"
        self._client = None
        self._setup_provider()

    def _setup_provider(self):
        """Initialize the LLM provider."""
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "ollama":
            self._setup_ollama()
        elif self.provider == "huggingface":
            self._setup_huggingface()
        else:
            logger.warning(f"Unknown LLM provider: {self.provider}, using fallback")
            self.provider = "fallback"

    def _setup_openai(self):
        """Setup OpenAI client."""
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. Generation will use fallback.")
            self.provider = "fallback"
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"✅ OpenAI client initialized (model: {OPENAI_MODEL})")
        except ImportError:
            logger.warning("openai package not installed")
            self.provider = "fallback"
        except Exception as e:
            logger.error(f"OpenAI setup failed: {e}")
            self.provider = "fallback"

    def _setup_ollama(self):
        """Setup Ollama client."""
        try:
            import httpx
            # Test connection
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
            if resp.status_code == 200:
                logger.info(f"✅ Ollama connected (model: {OLLAMA_MODEL})")
            else:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Using fallback.")
            self.provider = "fallback"

    def _setup_huggingface(self):
        """Setup HuggingFace model."""
        try:
            from transformers import pipeline
            import torch
            
            # Use TinyLlama as a small, fast local model for Mac
            local_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logger.info(f"⏳ Loading local model {local_model}...")
            
            self._client = pipeline(
                "text-generation",
                model=local_model,
                device_map="auto",
                max_new_tokens=256,
                temperature=0.1,
            )
            logger.info(f"✅ HuggingFace model loaded: {local_model}")
        except Exception as e:
            logger.warning(f"HuggingFace setup failed: {e}. Using fallback.")
            self.provider = "fallback"

    def generate(
        self,
        query: str,
        passages: list[dict],
        intent: str = "general",
        entities: list[dict] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Extract the most relevant sentences from passages instead of using an LLM.
        Returns a hardcoded JSON string that is parsed by the frontend to provide
        the 4 sections: Simple Answer, Supporting Text, Key Entities, Confidence.
        """
        if not passages:
            return json.dumps({
                "simple_answer": "No relevant passages were found to answer this question.",
                "supporting_text": "",
                "key_entities": [],
                "confidence": 0.0
            })

        logger.debug(f"Extracting answer for query: {query}")
        
        # Best chunk is the top ranked one
        best_passage = passages[0]
        text = best_passage.get("text", "")
        passage_entities = best_passage.get("entities", [])
        
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Simple scoring: Jaccard similarity between query words and sentence words
        query_words = set(query.lower().split())
        scored_sentences = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            if not query_words or not sent_words:
                score = 0
            else:
                intersection = len(query_words.intersection(sent_words))
                union = len(query_words.union(sent_words))
                score = intersection / union
            scored_sentences.append((score, sent))
            
        # Sort and take top 1 to 2 sentences to form simple answer
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Select best 2 sentences (if available) and maintain their original order
        best_sents = [s for _, s in scored_sentences[:2]]
        ordered_best = [s for s in sentences if s in best_sents]
        simple_answer = " ".join(ordered_best)
        
        if not simple_answer.strip():
            simple_answer = "Could not extract a specific sentence, but the document may be relevant."

        # Grab key entities from the passage to display
        key_entities = list(set([e.get("text", "") + f" ({e.get('label', '')})" for e in passage_entities if isinstance(e, dict)]))
        
        # Calculate a normalized confidence score [0, 1.0] from arbitrary retrieval scores
        raw_score = best_passage.get("score", 0.0)
        
        # Normalize: if negative, map it cleanly using sigmoid or logistic function to 0-1, or just max(0, min(1, score)) for simple scaling if it was normalized
        # Since FAISS/Cross-Encoder scores can be negative logits (-10 to 10 typical):
        normalized_confidence = 1 / (1 + math.exp(-raw_score)) if raw_score < 0 or raw_score > 1 else raw_score
            
        confidence_rounded = round(normalized_confidence, 2)

        result_dict = {
            "simple_answer": simple_answer,
            "supporting_text": text,
            "key_entities": key_entities,
            "confidence": confidence_rounded
        }

        # We return a JSON string which the frontend (or router) will parse.
        return json.dumps(result_dict)

    def _generate_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using OpenAI API."""
        try:
            response = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback_from_prompt(user)

    def _generate_ollama(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using Ollama API."""
        try:
            import httpx
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": system,
                    "prompt": user,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                    "stream": False,
                },
                timeout=120,
            )
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._generate_fallback_from_prompt(user)

    def _generate_huggingface(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate using HuggingFace model."""
        try:
            prompt = f"{system}\n\n{user}"
            result = self._client(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            generated = result[0]["generated_text"]
            # Remove the prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._generate_fallback_from_prompt(user)

    def _generate_fallback(
        self, query: str, passages: list[dict], intent: str
    ) -> str:
        """
        Fallback generation: extractive answer from passages.
        Used when no LLM is available.
        """
        logger.info("Using extractive fallback (no LLM available)")

        answer_parts = []
        answer_parts.append(f"Based on the retrieved documents, here is the relevant information:\n")

        for i, passage in enumerate(passages[:3], 1):
            text = passage.get("text", "")
            meta = passage.get("metadata", {})
            source = meta.get("doc_name", "Unknown")

            # Extract most relevant sentences
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            # Take first 3-5 sentences
            relevant = ". ".join(sentences[:5])
            if relevant and not relevant.endswith("."):
                relevant += "."

            answer_parts.append(
                f"**[Passage {i} — {source}]**: {relevant}"
            )

        answer_parts.append(
            "\n*Note: This is an extractive answer. Configure an LLM "
            "(OpenAI/Ollama) for generated answers with better synthesis.*"
        )

        return "\n\n".join(answer_parts)

    def _generate_fallback_from_prompt(self, prompt: str) -> str:
        """Extract an answer from passages within the prompt."""
        # Try to extract passage content from the prompt
        lines = prompt.split("\n")
        passage_lines = [l for l in lines if l.strip() and not l.startswith("---")]

        return (
            "I was unable to generate a synthesized answer due to LLM connectivity issues. "
            "Please refer to the source passages below for relevant information."
        )
