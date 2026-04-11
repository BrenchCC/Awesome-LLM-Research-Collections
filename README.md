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
  - [Agentic RL](#agentic-rl)
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
- **Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding** (2026.04) \
  **Description**: This paper introduces Video-MME-v2, an improved video understanding benchmark addressing the saturation issue in existing benchmarks where inflated leaderboard scores fail to reflect real-world model capabilities. \
  [[Paper](https://arxiv.org/abs/2604.05015)]
  [[Project](https://video-mme-v2.netlify.app/)]
  [[Code](https://github.com/MME-Benchmarks/Video-MME-v2)]
  [[Hugging Face](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2)]

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

- **InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency** (2025.08) \
  **Description**: This paper introduces InternVL 3.5, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency, featuring the Cascade Reinforcement Learning framework. \
  [[Paper](https://arxiv.org/abs/2508.18265)]

- **Qwen2.5-VL Technical Report** (2025.02) \
  **Description**: This technical report introduces Qwen2.5-VL, a flagship vision-language model with stronger visual recognition, precise localization, robust document parsing, and long-video understanding. It also improves agentic interaction with visual environments through better grounding and structured perception capabilities. \
  [[Paper](https://arxiv.org/abs/2502.13923)]
  [[Code](https://github.com/QwenLM/Qwen2.5-VL)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen25-vl-67ad1d2357b3028d1e9c4d56)]

- **InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling** (2024.12) \
  **Description**: This paper introduces InternVL 2.5, an advanced multimodal LLM series that was the first open-source MLLMs to surpass 70% on the MMMU benchmark, achieving a 3.7-point improvement through Chain-of-Thought reasoning. \
  [[Paper](https://arxiv.org/abs/2412.05271)]
  [[Hugging Face](https://huggingface.co/spaces/OpenGVLab/InternVL)]

- **LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models** (2024.07) \
  **Description**: This paper introduces LLaVA-NeXT-Interleave, which simultaneously tackles Multi-image, Multi-frame (video), Multi-view (3D), and Multi-patch scenarios in large multimodal models, extending visual instruction tuning to multi-modal scenarios. \
  [[Paper](https://arxiv.org/abs/2407.07895)]
  [[Project](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)]
  [[Code](https://github.com/LLaVA-VL/LLaVA-NeXT)]

- **How Good is my Video LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs** (2024.05) \
  **Description**: This paper introduces CVRR-ES, a benchmark that comprehensively assesses Video-LMMs across 11 diverse real-world video dimensions, evaluating 9 recent models and finding that most open-source Video-LMMs struggle with robustness and reasoning on complex videos. \
  [[Paper](https://arxiv.org/abs/2405.03690)]
  [[Project](https://mbzuai-oryx.github.io/CVRR-Evaluation-Suite/)]

- **LLaVA: Visual Instruction Tuning** (2023.04) \
  **Description**: This paper presents LLaVA, a large multimodal model trained end-to-end on machine-generated instruction tuning data, showing impressive multimodal chat abilities and achieving state-of-the-art results on Science QA. \
  [[Paper](https://arxiv.org/abs/2304.08485)]
  [[Project](https://llava-vl.github.io/)]

# Embeddings

# Training

## SFT
- **ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection** (2026.01) \
  **Description**: This paper presents ProFit, a supervised fine-tuning method that mitigates single-reference overfitting by using token probability as a proxy for semantic importance and masking low-probability tokens. The approach focuses learning on core logical content and improves reasoning and math performance over standard SFT baselines. \
  [[Paper](https://arxiv.org/abs/2601.09195)]

# Reinforcement Learning

- **Self-Distilled RLVR** (2026.04) \
  **Description**: This paper introduces Self-Distilled RLVR, combining on-policy distillation (OPD) with RLVR, where a larger teacher model provides dense fine-grained signals for each sampled trajectory, addressing the sparse signal limitation of standard RLVR. \
  [[Paper](https://arxiv.org/abs/2604.03128)]

- **The Art of Efficient Reasoning: Data, Reward, and Optimization** (2026.03) \
  **Description**: This paper studies efficient reasoning in LLMs, using RL to incentivize short accurate trajectories, with findings on training stages, rewards, and generalization across models from 0.6B to 30B parameters. \
  [[Paper](https://arxiv.org/abs/2602.20945)]
  [[Project](https://wutaiqiang.github.io/project/Art)]

- **Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis** (2026.02) \
  **Description**: This paper introduces Agentic Proposing, a framework that uses a specialized agent with Multi-Granularity Policy Optimization (MGPO) to dynamically select and compose modular reasoning skills for synthesizing high-precision training trajectories. \
  [[Paper](https://arxiv.org/abs/2602.03279)]

- **Reward Modeling from Natural Language Human Feedback** (2026.01) \
  **Description**: This paper introduces RLVR on preference data for training Generative Reward Models, demonstrating that binary classification tasks make GRMs susceptible to guessing correct outcomes without sound critiques, and proposes a method to address this limitation. \
  [[Paper](https://arxiv.org/abs/2601.07349)]

## Reasoning RL
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025.01) \
  **Description**: This paper shows that pure reinforcement learning can directly elicit advanced reasoning behaviors in LLMs without human-labeled reasoning traces. The proposed framework induces self-reflection, verification, and adaptive strategy use, leading to strong gains on math, coding, and STEM reasoning tasks. \
  [[Paper](https://arxiv.org/abs/2501.12948)]
  [[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)]

## Agentic RL
- **Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration** (2026.02) \
  **Description**: This paper introduces Actor-Refiner collaboration to address the multi-scale credit assignment problem in search-integrated reasoning RL, where sparse trajectory-level rewards fail to distinguish high-quality reasoning from fortuitous guesses, reducing redundant or misleading search behaviors. \
  [[Paper](https://arxiv.org/abs/2602.03647)]

- **Arena-RL: Training LLMs as Game Players with Vision-Language Action Models** (2026.01) \
  **Description**: This paper introduces Arena-RL, a reinforcement learning framework that trains LLM-driven agents to play visual games via vision-language action models, focusing on policy improvement from interactive game feedback. It demonstrates that reward-driven optimization over game trajectories can significantly improve strategic decision-making and generalization across game environments. \
  [[Paper](https://arxiv.org/abs/2601.06487)]
  
- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025.03) \
  **Description**: This paper introduces Search-R1, an RL framework where LLMs learn to autonomously generate search queries during step-by-step reasoning with real-time retrieval, improving their ability to acquire external knowledge and up-to-date information. \
  [[Paper](https://arxiv.org/abs/2503.09516)]
  [[Code](https://github.com/PeterGriffinJin/Search-R1)]

- **Search-o1: Agentic Search-Enhanced Large Reasoning Models** (2025.01) \
  **Description**: This paper introduces Search-o1, a framework that enhances large reasoning models with an agentic retrieval-augmented generation mechanism and a Reason-in-Documents module for refining retrieved documents, addressing knowledge insufficiency in extended reasoning processes. \
  [[Paper](https://arxiv.org/abs/2501.05366)]
  [[Code](https://github.com/sunnynexus/Search-o1)]

## VLA RL
- **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models** (2025.11) \
  **Description**: This paper proposes SRPO, a reinforcement learning framework for vision-language-action models that replaces sparse binary rewards with progress-wise rewards derived from the model's own successful trajectories. It uses latent world-model representations to measure behavioral progress robustly and achieves state-of-the-art manipulation success on LIBERO with far fewer RL steps. \
  [[Paper](https://arxiv.org/abs/2511.15605)]

# Agents Application

## Memory
- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (2025.10) \
  **Description**: This paper introduces BEAM, a benchmark of long, coherent conversations and probing questions for evaluating long-term memory in LLMs, and proposes LIGHT, a memory framework with episodic memory, working memory, and a scratchpad. Together, they expose long-context memory limitations and improve performance on long-horizon conversational reasoning tasks. \
  [[Paper](https://arxiv.org/abs/2510.27246)]]

- **MLP Memory: A Retriever-Pretrained Memory for Large Language Models** (2025.08) \
  **Description**: This paper introduces MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns by pretraining an MLP to imitate a kNN retriever's behavior, bridging the gap between RAG and fine-tuning approaches. \
  [[Paper](https://arxiv.org/abs/2508.01832)]

- **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** (2025.07) \
  **Description**: This paper introduces MemAgent, a multi-conversation RL-based memory agent that addresses the challenge of handling infinitely long documents with linear complexity without performance degradation during extrapolation. \
  [[Paper](https://arxiv.org/abs/2507.02259)]
  [[Project](https://memagent-sialab.github.io/)]

# Auto-Prompt

- **Becoming Experienced Judges: Selective Test-Time Learning for Evaluators** (2025.12) \
  **Description**: This paper introduces Learning While Evaluating (LWE), enabling LLM-as-a-judge systems to improve sequentially at inference time by updating an evolving meta-prompt with self-generated feedback. It further proposes Selective LWE, which updates only on self-inconsistent cases to improve evaluation quality with better cost efficiency. \
  [[Paper](https://arxiv.org/abs/2512.06751)]

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **Description**: APE improves LLM-as-a-judge reliability by automatically discovering auxiliary evaluation dimensions from failure cases and ensembling them with confidence-aware selection. It boosts agreement with human-aligned benchmarks by using test-time computation more effectively. \
  [[Paper](https://arxiv.org/abs/2510.06538)]
