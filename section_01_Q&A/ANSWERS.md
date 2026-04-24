# Task 1


Scenario
A customer support chatbot powered by GPT-4o was working well in testing. After going live,
users report three problems:

# Problem 1. (hallucinated pricing)

(1) the bot confidently gives wrong answers about product pricing

Investigation: Checked prompt templates, retrieval logs (RAG), and model temperature.

Root Cause: Retrieval/Knowledge Cutoff issue. The model is likely relying on pre-training data or outdated context rather than the live pricing database.

Fix: Implement a strict Grounding/RAG check where the bot is forced to query a real-time pricing API and instructed to refuse answers if the specific price isn't in the retrieved context

# Problem 2. (language switching)

(2) it occasionally responds in English even when the user writes in Hindi or Arabic

Investigation: Analyzed the system prompt and user message interaction.
+1

Root Cause: System Prompt Leakage/Weak Constraints. The model’s default training biases it toward English when it "forgets" the system instruction under high user message complexity.

Fix: Use a specific Language Anchor in the system prompt: "Always respond in the exact language used by the user. [Output: Hindi/Arabic/English]

# Problem 3. (latency degradation)

(3) response times have degraded from 1.2s to 8–12s over two weeks as the user base grew

Investigation: Checked vector store search time, API rate limits, and context window history.

Root Cause: Context Window Bloat. As the user base grew, conversation history likely exceeded optimal token limits, forcing the model to process massive amounts of past data for every turn.

Fix: Implement Conversation Summarization or a "sliding window" to cap the history sent to the API


# Post-Mortem Summary (Stakeholder Report)

Over the last two weeks,  we identified three performance bottlenecks. First, the bot gave incorrect prices because it relied on its own internal memory instead of our live database; we are fixing this by forcing it to "check the facts" before speaking. Second, it occasionally spoke English to non-English users due to weak language instructions; we have now reinforced its "mirroring" capability. Finally, the slowdown from 1s to 12s was caused by the bot trying to remember too much history at once; we are moving to a summarized memory system to keep responses lightning-fast as we scale