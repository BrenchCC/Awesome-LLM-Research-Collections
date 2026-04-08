# Awesome-LLM-Research-Collections

> Tool Recommend for read paper faster: [幻觉翻译](https://hjfy.top/)

# Contents
- [Attention](#attention)
- [LLMs](#llms)
  - [Foundation Models](#foundation-models)
- [Multimodal LLMs](#multimodal-llms)
  - [Vision-Language](#vision-language)
- [Embeddings](#embeddings)
- [Training](#training)
  - [SFT](#sft)
- [Reinforcement Learning](#reinforcement-learning)
  - [Reasoning RL](#reasoning-rl)
  - [VLA RL](#vla-rl)
- [Agents Application](#agents-application)
  - [Memory](#memory)
- [Auto-Prompt](#auto-prompt)

# Attention
- **Attention Residuals** (2026.03) \
  **Description**: This work replaces fixed residual accumulation with attention over previous layer outputs, enabling input-dependent depth-wise aggregation and reducing PreNorm-induced representation dilution. It also introduces Block AttnRes for scalable training with lower memory and communication overhead. \
  [[Paper](https://arxiv.org/abs/2603.15031)]
  [[Project](https://github.com/MoonshotAI/Attention-Residuals)]

# LLMs

## Foundation Models
- **GLM-5: from Vibe Coding to Agentic Engineering** (2026.02) \
  **Description**: GLM-5 is a next-generation foundation model targeting long-horizon agentic engineering, with reduced training and inference cost and preserved long-context capability. It introduces asynchronous RL infrastructure and agent RL algorithms to improve post-training efficiency and real-world coding performance. \
  [[Paper](https://arxiv.org/abs/2602.15763)]
  [[Project](https://z.ai/blog/glm-5)]
  [[Code](https://github.com/zai-org/GLM-5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-5)]

- **Kimi K2.5: Visual Agentic Intelligence** (2026.02) \
  **Description**: This paper introduces an open-source multimodal agentic model that jointly optimizes text and vision through unified pretraining, SFT, and reinforcement learning. It also proposes Agent Swarm, a parallel orchestration framework for decomposing and executing complex tasks with coordinated agents. \
  [[Paper](https://arxiv.org/abs/2602.02276)]
  [[Project](https://www.kimi.com/ai-models/kimi-k2-5)]
  [[Code](https://github.com/MoonshotAI/Kimi-K2.5)]
  [[Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.5)]

- **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models** (2025.08) \
  **Description**: GLM-4.5 introduces an open-source MoE foundation model with hybrid reasoning modes (thinking/direct response) to better support agentic, reasoning, and coding tasks. It combines large-scale pretraining and RL-based post-training, and releases both full and compact variants with strong benchmark performance. \
  [[Paper](https://arxiv.org/abs/2508.06471)]
  [[Code](https://github.com/zai-org/GLM-4.5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-4.5)]

- **Kimi K2: Open Agentic Intelligence** (2025.07) \
  **Description**: Kimi K2 presents a trillion-parameter MoE language model focused on strong agentic, reasoning, and coding capabilities with stable large-scale training. The work introduces MuonClip with QK-clip to improve optimization stability and token efficiency during pretraining. \
  [[Paper](https://arxiv.org/abs/2507.20534)]
  [[Project](https://moonshotai.github.io/Kimi-K2/)]
  [[Code](https://github.com/MoonshotAI/Kimi-K2)]
  [[Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Base)]

- **Qwen3 Technical Report** (2025.05) \
  **Description**: This report presents the Qwen3 family spanning dense and MoE models across a wide parameter range, emphasizing stronger multilingual performance and efficiency. It unifies deliberative thinking and fast response modes in one framework and scales post-training to improve reasoning, coding, and agentic behavior. \
  [[Paper](https://arxiv.org/abs/2505.09388)]
  [[Project](https://qwen.ai/blog?id=qwen3)]
  [[Code](https://github.com/QwenLM/Qwen3)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen3)]

- **MiniMax-01: Scaling Foundation Models with Lightning Attention** (2025.01) \
  **Description**: MiniMax-01 introduces a long-context model family built around Lightning Attention and MoE to improve scaling efficiency and practical throughput. It combines optimized parallelization and communication-computation overlap to train large models with stronger long-context performance. \
  [[Paper](https://arxiv.org/abs/2501.08313)]
  [[Project](https://www.minimax.io/news/minimax-01-series-2)]
  [[Code](https://github.com/MiniMax-AI/MiniMax-01)]
  [[Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)]

- **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** (2024.09) \
  **Description**: This paper presents Qwen2.5-Math, a family of math-specialized language models that applies self-improvement throughout pre-training, post-training, and inference. The approach strengthens mathematical reasoning and tool-augmented problem solving across multiple model sizes. \
  [[Paper](https://arxiv.org/abs/2409.12122)]
  [[Code](https://github.com/QwenLM/Qwen2.5-Math)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen25-math-66f7162f2be749b8a8e63c8a)]

- **Qwen2 Technical Report** (2024.07) \
  **Description**: This report introduces the Qwen2 series of dense and mixture-of-experts language models, covering base and instruction-tuned variants across a broad parameter range. It emphasizes stronger multilingual, coding, math, and reasoning performance while remaining competitive with proprietary systems. \
  [[Paper](https://arxiv.org/abs/2407.10671)]
  [[Code](https://github.com/QwenLM/Qwen2)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6641dc1d5fbb1a48c8708a52)]

# Multimodal LLMs

## Vision-Language
- **V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators** (2026.04) \
  **Description**: V-Reflection converts MLLMs from passive visual consumers to active interrogators through a think-then-look reflection mechanism that grounds each reasoning step in visual evidence. A two-stage distillation design improves fine-grained perception while keeping inference fully autoregressive and efficient. \
  [[Paper](https://arxiv.org/abs/2604.03307)]
  [[Project](https://idea-research.github.io/V-Reflection/)]
  [[Code](https://github.com/IDEA-Research/V-Reflection)]
  [[Hugging Face](https://huggingface.co/garlandchou/V-Reflection)]]

- **Qwen3-VL Technical Report** (2025.11) \
  **Description**: We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. \
  [[Paper](https://arxiv.org/abs/2511.21631)]
  [[Project](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list)]
  [[Code](https://github.com/QwenLM/Qwen3-VL)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen3-vl)]]

- **Qwen2.5-VL Technical Report** (2025.02) \
  **Description**: This technical report introduces Qwen2.5-VL, a flagship vision-language model with stronger visual recognition, precise localization, robust document parsing, and long-video understanding. It also improves agentic interaction with visual environments through better grounding and structured perception capabilities. \
  [[Paper](https://arxiv.org/abs/2502.13923)]
  [[Code](https://github.com/QwenLM/Qwen2.5-VL)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen25-vl-67ad1d2357b3028d1e9c4d56)]

# Embeddings

# Training

## SFT
- **ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection** (2026.01) \
  **Description**: This paper presents ProFit, a supervised fine-tuning method that mitigates single-reference overfitting by using token probability as a proxy for semantic importance and masking low-probability tokens. The approach focuses learning on core logical content and improves reasoning and math performance over standard SFT baselines. \
  [[Paper](https://arxiv.org/abs/2601.09195)]

# Reinforcement Learning

## Reasoning RL
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025.01) \
  **Description**: This paper shows that pure reinforcement learning can directly elicit advanced reasoning behaviors in LLMs without human-labeled reasoning traces. The proposed framework induces self-reflection, verification, and adaptive strategy use, leading to strong gains on math, coding, and STEM reasoning tasks. \
  [[Paper](https://arxiv.org/abs/2501.12948)]
  [[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)]

## VLA RL
- **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models** (2025.11) \
  **Description**: This paper proposes SRPO, a reinforcement learning framework for vision-language-action models that replaces sparse binary rewards with progress-wise rewards derived from the model's own successful trajectories. It uses latent world-model representations to measure behavioral progress robustly and achieves state-of-the-art manipulation success on LIBERO with far fewer RL steps. \
  [[Paper](https://arxiv.org/abs/2511.15605)]

# Agents Application

## Memory

- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (2025.10) \
  **Description**: This paper introduces BEAM, a benchmark of long, coherent conversations and probing questions for evaluating long-term memory in LLMs, and proposes LIGHT, a memory framework with episodic memory, working memory, and a scratchpad. Together, they expose long-context memory limitations and improve performance on long-horizon conversational reasoning tasks. \
  [[Paper](https://arxiv.org/abs/2510.27246)]]

# Auto-Prompt

- **Becoming Experienced Judges: Selective Test-Time Learning for Evaluators** (2025.12) \
  **Description**: This paper introduces Learning While Evaluating (LWE), enabling LLM-as-a-judge systems to improve sequentially at inference time by updating an evolving meta-prompt with self-generated feedback. It further proposes Selective LWE, which updates only on self-inconsistent cases to improve evaluation quality with better cost efficiency. \
  [[Paper](https://arxiv.org/abs/2512.06751)]

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **Description**: APE improves LLM-as-a-judge reliability by automatically discovering auxiliary evaluation dimensions from failure cases and ensembling them with confidence-aware selection. It boosts agreement with human-aligned benchmarks by using test-time computation more effectively. \
  [[Paper](https://arxiv.org/abs/2510.06538)]
