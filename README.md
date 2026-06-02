# Awesome-LLM-Research-Collections

> Language: English | [中文](README.zh-CN.md)

> Tool Recommend for read paper faster: [幻觉翻译](https://hjfy.top/)

# Contents
- [Attention](#attention)
  - [Attention Architecture](#attention-architecture)
- [LLMs](#llms)
  - [Foundation Models](#foundation-models)
  - [Inference](#inference)
  - [Detection](#detection)
- [Multimodal LLMs](#multimodal-llms)
  - [Vision-Language](#vision-language)
  - [Multimodal Reasoning](#multimodal-reasoning)
  - [VLA](#vla)
- [Embeddings](#embeddings)
- [SFT](#sft)
  - [SFT Methods](#sft-methods)
- [Training](#training)
  - [Optimization](#optimization)
  - [Distillation](#distillation)
- [Reinforcement Learning](#reinforcement-learning)
  - [OPD](#opd)
  - [Reward Modeling](#reward-modeling)
  - [Video Generation RL](#video-generation-rl)
  - [Reasoning RL](#reasoning-rl)
  - [Agentic RL](#agentic-rl)
  - [VLA RL](#vla-rl)
- [Agents Application](#agents-application)
  - [Tool Use](#tool-use)
  - [AI Research](#ai-research)
  - [Agent Skills](#agent-skills)
  - [Agent Development](#agent-development)
  - [Memory](#memory)
- [Vision](#vision)
  - [Object Detection](#object-detection)
- [Auto-Prompt](#auto-prompt)
  - [Prompt Optimization](#prompt-optimization)
  - [Judge Prompting](#judge-prompting)
- [Notes](#notes)
  - [Paper Readings](#paper-readings)
  - [Technical Reflections](#technical-reflections)

# Attention

## Attention Architecture
- **GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding** (2026.05) \
  **Description**: This paper proposes Group-Query Latent Attention, a minimal MLA modification that exposes both MQA-absorb and GQA decoding paths from the same trained weights. The runtime can select the path that matches target hardware without retraining or custom kernels, enabling H100-style compressed decoding, H20-oriented GQA plus MTP, and up to 8-way zero-redundancy tensor parallelism. \
  <a href="https://arxiv.org/abs/2605.15250"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/MuLabPKU/TransArch/tree/main/GQLA_preprint"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **Attention Residuals** (2026.03) \
  **Description**: This work replaces fixed residual accumulation with attention over previous layer outputs, enabling input-dependent depth-wise aggregation and reducing PreNorm-induced representation dilution. It also introduces Block AttnRes for scalable training with lower memory and communication overhead. \
  <a href="https://arxiv.org/abs/2603.15031"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/MoonshotAI/Attention-Residuals"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

# LLMs

## Foundation Models
- **The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook** (2026.04) \
  **Description**: This survey argues that continuous latent space is becoming a native computational substrate for language-based models, addressing the inefficiencies of explicit token-level generation such as redundancy, discretization bottlenecks, and semantic loss. It further organizes the field through mechanism and ability perspectives, and outlines key open challenges for future research. \
  <a href="https://arxiv.org/abs/2604.02029"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/YU-deep/Awesome-Latent-Space"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

- **GLM-5: from Vibe Coding to Agentic Engineering** (2026.02) \
  **Description**: GLM-5 is a next-generation foundation model targeting long-horizon agentic engineering, with reduced training and inference cost and preserved long-context capability. It introduces asynchronous RL infrastructure and agent RL algorithms to improve post-training efficiency and real-world coding performance. \
  <a href="https://arxiv.org/abs/2602.15763"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://z.ai/blog/glm-5"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/zai-org/GLM-5"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/zai-org/GLM-5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Kimi K2.5: Visual Agentic Intelligence** (2026.02) \
  **Description**: This paper introduces an open-source multimodal agentic model that jointly optimizes text and vision through unified pretraining, SFT, and reinforcement learning. It also proposes Agent Swarm, a parallel orchestration framework for decomposing and executing complex tasks with coordinated agents. \
  <a href="https://arxiv.org/abs/2602.02276"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://www.kimi.com/ai-models/kimi-k2-5"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/MoonshotAI/Kimi-K2.5"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/moonshotai/Kimi-K2.5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **MiMo-V2-Flash Technical Report** (2026.01) \
  **Description**: MiMo-V2-Flash is a 309B-parameter MoE foundation model with 15B active parameters, built for fast reasoning, coding, and agentic workloads through hybrid sliding-window/global attention, 27T-token pretraining, and long-context extension to 256k. It introduces Multi-Teacher On-Policy Distillation for scalable post-training and repurposes multi-token prediction as a draft model for speculative decoding speedups. \
  <a href="https://arxiv.org/abs/2601.02780"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/xiaomimimo/MiMo-V2-Flash"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models** (2025.12) \
  **Description**: DeepSeek-V3.2 is an open large language model that combines efficient long-context computation with strong reasoning and agent performance. Its key ingredients include DeepSeek Sparse Attention, scalable RL post-training, and a large-scale agentic task synthesis pipeline for improving tool-use generalization and instruction-following robustness. \
  <a href="https://arxiv.org/abs/2512.02556"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://api-docs.deepseek.com/news/news251201"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models** (2025.08) \
  **Description**: GLM-4.5 introduces an open-source MoE foundation model with hybrid reasoning modes (thinking/direct response) to better support agentic, reasoning, and coding tasks. It combines large-scale pretraining and RL-based post-training, and releases both full and compact variants with strong benchmark performance. \
  <a href="https://arxiv.org/abs/2508.06471"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/zai-org/GLM-4.5"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/zai-org/GLM-4.5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Kimi K2: Open Agentic Intelligence** (2025.07) \
  **Description**: Kimi K2 presents a trillion-parameter MoE language model focused on strong agentic, reasoning, and coding capabilities with stable large-scale training. The work introduces MuonClip with QK-clip to improve optimization stability and token efficiency during pretraining. \
  <a href="https://arxiv.org/abs/2507.20534"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://moonshotai.github.io/Kimi-K2/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/MoonshotAI/Kimi-K2"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/moonshotai/Kimi-K2-Base"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen3 Technical Report** (2025.05) \
  **Description**: This report presents the Qwen3 family spanning dense and MoE models across a wide parameter range, emphasizing stronger multilingual performance and efficiency. It unifies deliberative thinking and fast response modes in one framework and scales post-training to improve reasoning, coding, and agentic behavior. \
  <a href="https://arxiv.org/abs/2505.09388"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://qwen.ai/blog?id=qwen3"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen3"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen3"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **MiniMax-01: Scaling Foundation Models with Lightning Attention** (2025.01) \
  **Description**: MiniMax-01 introduces a long-context model family built around Lightning Attention and MoE to improve scaling efficiency and practical throughput. It combines optimized parallelization and communication-computation overlap to train large models with stronger long-context performance. \
  <a href="https://arxiv.org/abs/2501.08313"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://www.minimax.io/news/minimax-01-series-2"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-01"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeek-V3 Technical Report** (2024.12) \
  **Description**: DeepSeek-V3 is a 671B-parameter MoE language model with 37B activated parameters per token, built for efficient inference and cost-effective large-scale training. It extends MLA and DeepSeekMoE with auxiliary-loss-free load balancing and a multi-token prediction objective, achieving strong open-model performance with stable 14.8T-token pretraining and SFT/RL post-training. \
  <a href="https://arxiv.org/abs/2412.19437"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** (2024.09) \
  **Description**: This paper presents Qwen2.5-Math, a family of math-specialized language models that applies self-improvement throughout pre-training, post-training, and inference. The approach strengthens mathematical reasoning and tool-augmented problem solving across multiple model sizes. \
  <a href="https://arxiv.org/abs/2409.12122"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2.5-Math"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen25-math-66f7162f2be749b8a8e63c8a"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen2 Technical Report** (2024.07) \
  **Description**: This report introduces the Qwen2 series of dense and mixture-of-experts language models, covering base and instruction-tuned variants across a broad parameter range. It emphasizes stronger multilingual, coding, math, and reasoning performance while remaining competitive with proprietary systems. \
  <a href="https://arxiv.org/abs/2407.10671"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen2-6641dc1d5fbb1a48c8708a52"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** (2024.05) \
  **Description**: DeepSeek-V2 is a 236B-parameter MoE language model with 21B activated parameters per token and 128K context length, designed for economical training and efficient inference. It combines Multi-head Latent Attention for KV-cache compression with DeepSeekMoE sparse computation, reducing training cost and KV cache while improving throughput and open-model performance. \
  <a href="https://arxiv.org/abs/2405.04434"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## Inference
- **Large Language Models Explore by Latent Distilling** (2026.04) \
  **Description**: This paper proposes Exploratory Sampling (ESamp), a decoding method that addresses the shallow lexical variation of standard stochastic sampling by encouraging semantic exploration. It trains a lightweight Distiller at test time to predict deep-layer representations from shallow ones, then uses prediction error as a novelty signal to reweight candidate tokens and improve Pass@k efficiency. \
  <a href="https://arxiv.org/abs/2604.24927"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/LinesHogan/tLLM"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **Caterpillar of Thoughts: The Optimal Test-Time Algorithm for Large Language Models** (2026.03) \
  **Description**: This paper presents a theoretical framework for optimal test-time computation in LLMs, proving that the optimal algorithm always generates a caterpillar tree structure, and introduces CaT which achieves better success rate than Tree-of-Thoughts with fewer token generations. \
  <a href="https://arxiv.org/abs/2603.22784"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Detection
- **Base Models Look Human To AI Detectors** (2026.05) \
  **Description**: This paper finds that commercial AI-text detectors often label base-model outputs as more human than outputs from instruction-tuned counterparts, suggesting they track tuning artifacts and local context rather than invariant machine-text signals. It proposes Humanization by Iterative Paraphrasing (HIP), a detector-agnostic fine-tuning and iterative paraphrasing pipeline that improves semantic preservation while evading detectors. \
  <a href="https://arxiv.org/abs/2605.19516"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

# Multimodal LLMs

## Vision-Language
- **Lance: Unified Multimodal Modeling by Multi-Task Synergy** (2026.05) \
  **Description**: Lance introduces a lightweight native unified multimodal model for image and video understanding, generation, and editing without relying mainly on capacity scaling. It combines shared interleaved context modeling, decoupled capability pathways, dual-stream MoE, modality-aware rotary positional encoding, and staged multi-task training to improve both generation and understanding. \
  <a href="https://arxiv.org/abs/2605.18678"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://lance-project.github.io/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/bytedance/Lance"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/bytedance-research/Lance"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding** (2026.04) \
  **Description**: This paper introduces Video-MME-v2, an improved video understanding benchmark addressing the saturation issue in existing benchmarks where inflated leaderboard scores fail to reflect real-world model capabilities. \
  <a href="https://arxiv.org/abs/2604.05015"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://video-mme-v2.netlify.app/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/MME-Benchmarks/Video-MME-v2"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators** (2026.04) \
  **Description**: V-Reflection converts MLLMs from passive visual consumers to active interrogators through a think-then-look reflection mechanism that grounds each reasoning step in visual evidence. A two-stage distillation design improves fine-grained perception while keeping inference fully autoregressive and efficient. \
  <a href="https://arxiv.org/abs/2604.03307"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://idea-research.github.io/V-Reflection/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/IDEA-Research/V-Reflection"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/garlandchou/V-Reflection"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model** (2026.03) \
  **Description**: This paper proposes a VLM-guided JEPA-style latent world modeling framework that combines dense-frame dynamics prediction with long-horizon semantic guidance through a dual-temporal design. It further introduces a hierarchical pyramid representation extraction module to transfer multi-layer VLM reasoning features into latent forecasting for more robust hand-manipulation trajectory prediction. \
  <a href="https://arxiv.org/abs/2603.22281"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Qwen3-VL Technical Report** (2025.11) \
  **Description**: We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. \
  <a href="https://arxiv.org/abs/2511.21631"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen3-VL"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen3-vl"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency** (2025.08) \
  **Description**: This paper introduces InternVL 3.5, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency, featuring the Cascade Reinforcement Learning framework. \
  <a href="https://arxiv.org/abs/2508.18265"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Qwen2.5-VL Technical Report** (2025.02) \
  **Description**: This technical report introduces Qwen2.5-VL, a flagship vision-language model with stronger visual recognition, precise localization, robust document parsing, and long-video understanding. It also improves agentic interaction with visual environments through better grounding and structured perception capabilities. \
  <a href="https://arxiv.org/abs/2502.13923"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2.5-VL"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen25-vl-67ad1d2357b3028d1e9c4d56"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling** (2024.12) \
  **Description**: This paper introduces InternVL 2.5, an advanced multimodal LLM series that was the first open-source MLLMs to surpass 70% on the MMMU benchmark, achieving a 3.7-point improvement through Chain-of-Thought reasoning. \
  <a href="https://arxiv.org/abs/2412.05271"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://huggingface.co/spaces/OpenGVLab/InternVL"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models** (2024.07) \
  **Description**: This paper introduces LLaVA-NeXT-Interleave, which simultaneously tackles Multi-image, Multi-frame (video), Multi-view (3D), and Multi-patch scenarios in large multimodal models, extending visual instruction tuning to multi-modal scenarios. \
  <a href="https://arxiv.org/abs/2407.07895"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/LLaVA-VL/LLaVA-NeXT"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **How Good is my Video LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs** (2024.05) \
  **Description**: This paper introduces CVRR-ES, a benchmark that comprehensively assesses Video-LMMs across 11 diverse real-world video dimensions, evaluating 9 recent models and finding that most open-source Video-LMMs struggle with robustness and reasoning on complex videos. \
  <a href="https://arxiv.org/abs/2405.03690"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://mbzuai-oryx.github.io/CVRR-Evaluation-Suite/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

- **LLaVA: Visual Instruction Tuning** (2023.04) \
  **Description**: This paper presents LLaVA, a large multimodal model trained end-to-end on machine-generated instruction tuning data, showing impressive multimodal chat abilities and achieving state-of-the-art results on Science QA. \
  <a href="https://arxiv.org/abs/2304.08485"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://llava-vl.github.io/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

## Multimodal Reasoning
- **Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey** (2025.03) \
  **Description**: This survey addresses the lack of an up-to-date review of multimodal Chain-of-Thought reasoning in MLLMs across image, video, speech, audio, 3D, and structured data. It introduces foundational definitions, a comprehensive taxonomy, methodological analysis across applications, and open challenges for future multimodal reasoning research. \
  <a href="https://arxiv.org/abs/2503.12605"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/yaotingwangofficial/Awesome-MCoT"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

## VLA
- **Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation** (2026.04) \
  **Description**: OneVL addresses real-time trajectory planning in VLA-based autonomous driving by compressing Chain-of-Thought reasoning into compact latent tokens supervised by both language reconstruction and future-frame prediction. Its three-stage training pipeline yields latent reasoning that surpasses explicit CoT while keeping answer-only inference latency. \
  <a href="https://arxiv.org/abs/2604.18486"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://xiaomi-embodied-intelligence.github.io/OneVL/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/xiaomi-research/onevl"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/xiaomi-research/onevl-models"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

# Embeddings

# SFT

## SFT Methods
- **Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability** (2026.04) \
  **Description**: This paper challenges the common claim that supervised fine-tuning (SFT) only memorizes while RL generalizes, finding cross-domain generalization from reasoning SFT with long chain-of-thought supervision depends jointly on optimization dynamics, training data, and base model capability. \
  <a href="https://arxiv.org/abs/2604.06628"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/Nebularaid2000/rethink_sft_generalization"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/collections/jasonrqh/rethink-sft-generalization"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection** (2026.01) \
  **Description**: This paper presents ProFit, a supervised fine-tuning method that mitigates single-reference overfitting by using token probability as a proxy for semantic importance and masking low-probability tokens. The approach focuses learning on core logical content and improves reasoning and math performance over standard SFT baselines. \
  <a href="https://arxiv.org/abs/2601.09195"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

# Training

## Optimization
- **PowLU: An Activation Function for Stable Pre-Training of LLMs** (2026.05) \
  **Description**: This paper identifies SwiGLU's near-quadratic amplification on large positive inputs as a source of outliers and numerical instability in low-precision large-scale LLM pre-training. It proposes Power Linear Unit (PowLU), a rational-power activation that preserves adaptive nonlinearity while stabilizing spike regions, with scaling-law and Ling-model experiments showing competitive performance and improved training scalability. \
  <a href="https://arxiv.org/abs/2605.25704"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Distillation
- **Knowledge Distillation of Large Language Models** (2023.06) \
  **Description**: This paper studies white-box knowledge distillation for generative LLMs and proposes MiniLLM, replacing the standard forward KLD objective with reverse KLD to avoid overestimating low-probability teacher regions. The method derives an effective optimization procedure and improves instruction-following quality, calibration, exposure bias, and long-text generation across model families from 120M to 13B parameters. \
  <a href="https://arxiv.org/abs/2306.08543"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/microsoft/LMOps/tree/main/minillm"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/MiniLLM"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

# Reinforcement Learning

## OPD
- **Self-Distilled RLVR** (2026.04) \
  **Description**: This paper studies on-policy self-distillation for RLVR and shows that relying only on a privileged self-teacher can cause information leakage and unstable long-term training. It proposes RLSD, which uses self-distillation to estimate token-level update magnitudes while keeping RLVR's environment feedback as the reliable update direction. \
  <a href="https://arxiv.org/abs/2604.03128"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** (2026.02) \
  **Description**: This paper shows that on-policy distillation is a special case of dense KL-constrained RL, then proposes G-OPD with a flexible reference model and reward scaling factor. Its reward extrapolation variant, ExOPD, improves over standard OPD and can let students surpass domain teachers when merging RL-trained experts. \
  <a href="https://arxiv.org/abs/2602.12125"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/RUCBM/G-OPD"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/datasets/Keven16/G-OPD-Training-Data"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## Reward Modeling
- **MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels** (2026.03) \
  **Description**: This paper introduces MemReward, a graph-based experience memory framework that achieves 97.3% of Oracle performance on 3B and 96.6% on 1.5B models for reward prediction with limited labels, surpassing Oracle in out-of-domain tasks. \
  <a href="https://arxiv.org/abs/2603.19310"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/ulab-uiuc/MemReward"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/datasets/ulab-ai/MemReward"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Scaling Reward Modeling without Human Supervision** (2026.03) \
  **Description**: This paper studies unsupervised reward model scaling by learning preferences over web-corpus document prefixes and suffixes without human annotations. It reports consistent RewardBench gains across model backbones and shows downstream improvements in best-of-N selection and policy optimization. \
  <a href="https://arxiv.org/abs/2603.02225"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Reward Modeling from Natural Language Human Feedback** (2026.01) \
  **Description**: This paper introduces RLVR on preference data for training Generative Reward Models, demonstrating that binary classification tasks make GRMs susceptible to guessing correct outcomes without sound critiques, and proposes a method to address this limitation. \
  <a href="https://arxiv.org/abs/2601.07349"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models** (2025.10) \
  **Description**: This survey reviews Process Reward Models for evaluating and guiding LLM reasoning at the step or trajectory level rather than only judging final answers. It organizes the full loop of process data generation, PRM construction, and PRM use in test-time scaling and reinforcement learning across math, code, multimodal reasoning, robotics, and agents. \
  <a href="https://arxiv.org/abs/2510.08049"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Video Generation RL
- **KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration** (2026.05) \
  **Description**: KVPO aligns streaming autoregressive video generators with human preferences using an ODE-native online GRPO framework. It replaces noise-based exploration with causal-semantic routing of historical KV cache entries and optimizes a velocity-field surrogate policy based on Trajectory Velocity Energy. \
  <a href="https://arxiv.org/abs/2605.14278"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://richard-zhang-ai.github.io/KVPO-Project/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/Richard-Zhang-AI/KVPO"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/Richard-ZZZZZ/KVPO"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## Reasoning RL
- **All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models** (2026.04) \
  **Description**: This paper presents MUPO, a reinforcement learning approach that addresses diversity collapse in GRPO-trained VLMs by incentivizing divergent thinking across multiple solutions, enabling deeper yet broader reasoning patterns. \
  <a href="https://arxiv.org/abs/2604.00479"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://xytian1008.github.io/MUPO/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/xytian1008/MUPO"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/xytian1008/MUPO-Thinker-7B"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **The Art of Efficient Reasoning: Data, Reward, and Optimization** (2026.03) \
  **Description**: This paper studies efficient reasoning in LLMs, using RL to incentivize short accurate trajectories, with findings on training stages, rewards, and generalization across models from 0.6B to 30B parameters. \
  <a href="https://arxiv.org/abs/2602.20945"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://wutaiqiang.github.io/project/Art"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

- **FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization** (2026.03) \
  **Description**: This paper presents FIPO, a reinforcement learning algorithm that overcomes reasoning bottlenecks in LLMs by addressing coarse-grained credit assignment in GRPO-style training, where outcome-based rewards fail to distinguish critical logical pivots from trivial tokens. \
  <a href="https://arxiv.org/abs/2603.19835"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis** (2026.02) \
  **Description**: This paper introduces Agentic Proposing, a framework that uses a specialized agent with Multi-Granularity Policy Optimization (MGPO) to dynamically select and compose modular reasoning skills for synthesizing high-precision training trajectories. \
  <a href="https://arxiv.org/abs/2602.03279"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025.01) \
  **Description**: This paper shows that pure reinforcement learning can directly elicit advanced reasoning behaviors in LLMs without human-labeled reasoning traces. The proposed framework induces self-reflection, verification, and adaptive strategy use, leading to strong gains on math, coding, and STEM reasoning tasks. \
  <a href="https://arxiv.org/abs/2501.12948"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## Agentic RL
- **Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning** (2026.05) \
  **Description**: Spreadsheet-RL is an RL fine-tuning framework for training specialized spreadsheet agents in a realistic Microsoft Excel environment, addressing complex multi-step workflows that prompting-based agents struggle with. It adds automated start-goal spreadsheet data collection, a multi-turn Spreadsheet Gym with sandboxed Excel tools, and a Domain-Spreadsheet benchmark to improve real-world spreadsheet automation. \
  <a href="https://arxiv.org/abs/2605.22642"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://spreadsheet-rl.github.io/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/Spreadsheet-RL/Spreadsheet-RL"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/datasets/Spreadsheet-RL/Spreadsheet-RL"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration** (2026.02) \
  **Description**: This paper introduces Actor-Refiner collaboration to address the multi-scale credit assignment problem in search-integrated reasoning RL, where sparse trajectory-level rewards fail to distinguish high-quality reasoning from fortuitous guesses, reducing redundant or misleading search behaviors. \
  <a href="https://arxiv.org/abs/2602.03647"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Arena-RL: Training LLMs as Game Players with Vision-Language Action Models** (2026.01) \
  **Description**: This paper introduces Arena-RL, a reinforcement learning framework that trains LLM-driven agents to play visual games via vision-language action models, focusing on policy improvement from interactive game feedback. It demonstrates that reward-driven optimization over game trajectories can significantly improve strategic decision-making and generalization across game environments. \
  <a href="https://arxiv.org/abs/2601.06487"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  
- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025.03) \
  **Description**: This paper introduces Search-R1, an RL framework where LLMs learn to autonomously generate search queries during step-by-step reasoning with real-time retrieval, improving their ability to acquire external knowledge and up-to-date information. \
  <a href="https://arxiv.org/abs/2503.09516"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/PeterGriffinJin/Search-R1"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **Search-o1: Agentic Search-Enhanced Large Reasoning Models** (2025.01) \
  **Description**: This paper introduces Search-o1, a framework that enhances large reasoning models with an agentic retrieval-augmented generation mechanism and a Reason-in-Documents module for refining retrieved documents, addressing knowledge insufficiency in extended reasoning processes. \
  <a href="https://arxiv.org/abs/2501.05366"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/sunnynexus/Search-o1"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

## VLA RL
- **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models** (2025.11) \
  **Description**: This paper proposes SRPO, a reinforcement learning framework for vision-language-action models that replaces sparse binary rewards with progress-wise rewards derived from the model's own successful trajectories. It uses latent world-model representations to measure behavioral progress robustly and achieves state-of-the-art manipulation success on LIBERO with far fewer RL steps. \
  <a href="https://arxiv.org/abs/2511.15605"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

# Agents Application

## Tool Use
- **Thinking with Programming Vision: Towards a Unified View for Thinking with Images** (2025.12) \
  **Description**: This paper identifies brittleness in current multimodal tool-using reasoning under simple image orientation changes and corruptions, and proposes CodeVision, a code-as-tool framework that lets models invoke arbitrary image operations through generated code. It combines SFT and RL with dense process rewards to improve multi-tool reasoning, execution efficiency, and error recovery on thinking-with-images tasks. \
  <a href="https://arxiv.org/abs/2512.03746"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/ByteDance-BandAI/CodeVision"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

## AI Research
- **AI for Auto-Research: Roadmap & User Guide** (2026.05) \
  **Description**: This survey analyzes AI-assisted research across creation, writing, validation, and dissemination, showing where automation is reliable and where autonomy still fails on novelty, experiments, and scientific judgment. It provides a lifecycle taxonomy, benchmark suite, tool inventory, design principles, and practitioner playbook for human-governed AI research workflows. \
  <a href="https://arxiv.org/abs/2605.18661"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://worldbench.github.io/awesome-ai-auto-research"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/worldbench/awesome-ai-auto-research"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **AIRA_2: Overcoming Bottlenecks in AI Research Agents** (2026.03) \
  **Description**: This paper introduces AIRA_2, an AI research agent architecture that addresses limited experiment throughput, noisy validation-based selection, and static single-turn operators. It combines asynchronous multi-GPU workers, Hidden Consistent Evaluation, and interactive ReAct agents to improve long-horizon research task performance. \
  <a href="https://arxiv.org/abs/2603.26499"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Agent Skills
- **SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution** (2026.05) \
  **Description**: SkillsVote addresses noisy and hard-to-govern agent trajectories by treating Agent Skills as reusable experience artifacts with collection, recommendation, attribution, and evolution controls. It profiles large-scale open-source skill corpora, recommends structured skill context before execution, and admits only evidence-gated successful discoveries to improve frozen agents without model updates. \
  <a href="https://arxiv.org/abs/2605.18401"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://skills.vote/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>
  <a href="https://github.com/MemTensor/skills-vote"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

- **From Context to Skills: Can Language Models Learn from Context Skillfully?** (2026.04) \
  **Description**: Ctx2Skill addresses context learning for long, dense contexts where manual skill annotation is costly and automated skill construction lacks external feedback. It uses a multi-agent self-play loop with Cross-time Replay to autonomously discover, refine, and select reusable natural-language skills that improve solving rates across language models. \
  <a href="https://arxiv.org/abs/2604.27660"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/S1s-Z/Ctx2Skill"><img src="assets/icons/github.svg" alt="Code" width="20"></a>
  <a href="https://huggingface.co/datasets/ssz1111/Ctx2Skill"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **SkillReducer: Optimizing LLM Agent Skills for Token Efficiency** (2026.03) \
  **Description**: This paper presents SkillReducer, a two-stage optimization framework that compresses LLM agent skills (pre-packaged instruction sets) by 48% for descriptions and 39% for body while improving functional quality by 2.8%, reducing token costs and attention dilution in agent contexts. \
  <a href="https://arxiv.org/abs/2603.29919"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Agent Development
- **Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization** (2026.03) \
  **Description**: This paper proposes Nurture-First Development, a paradigm for growing domain-expert agents through structured conversational interaction rather than fixed code-first or prompt-first construction. It formalizes a Knowledge Crystallization Cycle, Three-Layer Cognitive Architecture, Dual-Workspace Pattern, and Spiral Development Model for continuously converting tacit practitioner knowledge into reusable agent assets. \
  <a href="https://arxiv.org/abs/2603.10808"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

## Memory
- **Trajectory-Informed Memory Generation for Self-Improving Agent Systems** (2026.03) \
  **Description**: This paper presents a framework that extracts actionable learnings from LLM agent execution trajectories and retrieves them as contextual memory for future tasks. It combines trajectory intelligence extraction, decision attribution, contextual learning generation, and adaptive memory retrieval to improve AppWorld task completion, especially on complex scenarios. \
  <a href="https://arxiv.org/abs/2603.10600"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (2025.10) \
  **Description**: This paper introduces BEAM, a benchmark of long, coherent conversations and probing questions for evaluating long-term memory in LLMs, and proposes LIGHT, a memory framework with episodic memory, working memory, and a scratchpad. Together, they expose long-context memory limitations and improve performance on long-horizon conversational reasoning tasks. \
  <a href="https://arxiv.org/abs/2510.27246"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **MLP Memory: A Retriever-Pretrained Memory for Large Language Models** (2025.08) \
  **Description**: This paper introduces MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns by pretraining an MLP to imitate a kNN retriever's behavior, bridging the gap between RAG and fine-tuning approaches. \
  <a href="https://arxiv.org/abs/2508.01832"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** (2025.07) \
  **Description**: This paper introduces MemAgent, a multi-conversation RL-based memory agent that addresses the challenge of handling infinitely long documents with linear complexity without performance degradation during extrapolation. \
  <a href="https://arxiv.org/abs/2507.02259"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://memagent-sialab.github.io/"><img src="assets/icons/project.svg" alt="Project" width="20"></a>

# Vision

## Object Detection
- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** (2022.03) \
  **Description**: DINO improves DETR-like object detectors by introducing contrastive denoising training, mixed query selection for anchor initialization, and a look-forward-twice box prediction scheme, achieving state-of-the-art results on COCO with significantly reduced model and data requirements. \
  <a href="https://arxiv.org/abs/2203.03605"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/IDEA-Research/DINO"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

# Auto-Prompt

## Prompt Optimization
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025.07) \
  **Description**: GEPA introduces a prompt optimizer that uses natural language reflection to learn high-level rules from trial and error, outperforming GRPO by 6% on average with up to 35x fewer rollouts. It also beats MIPROv2 by over 10% and shows promising results as an inference-time search strategy for code optimization. \
  <a href="https://arxiv.org/abs/2507.19457"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>
  <a href="https://github.com/gepa-ai/gepa"><img src="assets/icons/github.svg" alt="Code" width="20"></a>

## Judge Prompting
- **Becoming Experienced Judges: Selective Test-Time Learning for Evaluators** (2025.12) \
  **Description**: This paper introduces Learning While Evaluating (LWE), enabling LLM-as-a-judge systems to improve sequentially at inference time by updating an evolving meta-prompt with self-generated feedback. It further proposes Selective LWE, which updates only on self-inconsistent cases to improve evaluation quality with better cost efficiency. \
  <a href="https://arxiv.org/abs/2512.06751"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **Description**: APE improves LLM-as-a-judge reliability by automatically discovering auxiliary evaluation dimensions from failure cases and ensembling them with confidence-aware selection. It boosts agreement with human-aligned benchmarks by using test-time computation more effectively. \
  <a href="https://arxiv.org/abs/2510.06538"><img src="assets/icons/arxiv.svg" alt="Paper" width="20"></a>

# Notes

## Paper Readings

No notes yet.

## Technical Reflections

- **OPD: Capability Integration Interface in Post-training** (2026-05-28) \
  **Description**: A technical reflection on how OPD becomes a capability integration interface in post-training through Qwen3, GLM-5, MiMo-V2, and DeepSeek-V4. \
  [[Note](notes/en/opd/post-training-opd.qmd)]
  [[中文](notes/zh/opd/post-training-opd.qmd)]
