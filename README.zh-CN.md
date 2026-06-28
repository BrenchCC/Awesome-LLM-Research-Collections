# Awesome LLM 研究论文合集

> 语言：[English](README.md) | 中文
>
> 快速阅读论文推荐工具：[幻觉翻译](https://hjfy.top/)

# 目录
- [注意力机制](#注意力机制)
  - [注意力架构](#注意力架构)
- [大语言模型](#大语言模型)
  - [基础模型](#基础模型)
  - [推理](#推理)
  - [检测](#检测)
- [多模态大模型](#多模态大模型)
  - [视觉语言](#视觉语言)
  - [多模态推理](#多模态推理)
  - [视觉-语言-动作](#视觉-语言-动作)
- [嵌入模型](#嵌入模型)
- [监督微调](#监督微调)
  - [监督微调方法](#监督微调方法)
- [训练](#训练)
  - [优化](#优化)
  - [蒸馏](#蒸馏)
- [强化学习](#强化学习)
  - [策略优化](#策略优化)
  - [OPD](#opd)
  - [奖励建模](#奖励建模)
  - [视频生成强化学习](#视频生成强化学习)
  - [多模态强化学习](#多模态强化学习)
  - [推理强化学习](#推理强化学习)
  - [智能体强化学习](#智能体强化学习)
  - [视觉-语言-动作强化学习](#视觉-语言-动作强化学习)
- [智能体应用](#智能体应用)
  - [工具调用](#工具调用)
  - [AI 研究](#ai-研究)
  - [智能体技能](#智能体技能)
  - [智能体开发](#智能体开发)
  - [记忆](#记忆)
- [视觉](#视觉)
  - [目标检测](#目标检测)
- [自动提示](#自动提示)
  - [提示优化](#提示优化)
  - [评测器提示](#评测器提示)
- [笔记](#笔记)
  - [论文解读](#论文解读)
  - [技术思考](#技术思考)
- [博客](#博客)

# 注意力机制

## 注意力架构
- **FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention** (2026.06) \
  **描述**: 该论文提出 Lookahead Sparse Attention，使用独立训练的神经记忆索引器预测未来上下文需求，仅在 GPU 上保留查询关键的 KV 分块。FlashMemory 将物理 KV 缓存占用平均压缩至完整上下文注意力的 13.5%，同时保持或略微提升长上下文准确率。 \
  <a href="https://arxiv.org/abs/2606.09079"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/libertywing/FlashMemory-Deepseek-V4"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/libertywing/FlashMemory-Deepseek-V4"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **MiniMax Sparse Attention** (2026.06) \
  **描述**: 该论文提出 MiniMax Sparse Attention，一种基于 GQA 的分块稀疏注意力机制，通过轻量级 Index Branch 为每个 GQA 组选择 Top-k KV 分块，再执行精确稀疏注意力。其协同设计的 GPU kernel 将稀疏性转化为百万 token 上下文中的实际加速，同时保持接近 dense GQA 的模型效果。 \
  <a href="https://arxiv.org/abs/2606.13392"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/MiniMax-AI/MSA"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-M3"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding** (2026.05) \
  **描述**: 该论文提出 Group-Query Latent Attention，对 MLA 做最小修改，使同一组训练权重同时暴露 MQA-absorb 和 GQA 两条解码路径。运行时可根据目标硬件选择路径，无需重新训练或自定义 kernel，从而兼顾 H100 式压缩解码、面向 H20 的 GQA 加 MTP，以及最高 8 路零冗余张量并行。 \
  <a href="https://arxiv.org/abs/2605.15250"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/MuLabPKU/TransArch/tree/main/GQLA_preprint"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Attention Residuals** (2026.03) \
  **描述**: 该工作用对前序层输出的注意力替代固定残差累积，使模型能够根据输入动态聚合不同深度的信息，并缓解 PreNorm 带来的表征稀释问题。论文还提出 Block AttnRes，在更低显存与通信开销下支持可扩展训练。 \
  <a href="https://arxiv.org/abs/2603.15031"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/MoonshotAI/Attention-Residuals"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

- **Fast Transformer Decoding: One Write-Head is All You Need** (2019.11) \
  **描述**: 该论文提出 multi-query attention，在不同注意力头之间共享 keys 和 values，以降低 Transformer 增量解码中的内存带宽开销。该变体显著提升解码速度，同时相较多头注意力基线仅带来轻微质量下降。 \
  <a href="https://arxiv.org/abs/1911.02150"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

# 大语言模型

## 基础模型
- **The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence** (2026.05) \
  **描述**: 该技术报告介绍 MiniMax-M2 系列，一组以较小激活参数规模面向真实智能体部署的 MoE 语言模型。它结合智能体驱动的可验证数据管线、Forge 智能体原生 RL 系统，以及 M2.7 中的早期自演进机制，提升代码、深度搜索、办公任务和推理表现。 \
  <a href="https://arxiv.org/abs/2605.26494"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://www.minimax.io/blog/minimax-m27"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-M2.7"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-M2.7"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook** (2026.04) \
  **描述**: 这篇综述认为连续潜在空间正在成为语言模型的原生计算基底，可缓解显式 token 生成中的冗余、离散化瓶颈和语义损失。论文从机制和能力两个视角梳理该领域，并总结未来关键挑战。 \
  <a href="https://arxiv.org/abs/2604.02029"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/YU-deep/Awesome-Latent-Space"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

- **GLM-5: from Vibe Coding to Agentic Engineering** (2026.02) \
  **描述**: GLM-5 是面向长程智能体工程的新一代基础模型，在降低训练与推理成本的同时保持长上下文能力。它引入异步 RL 基础设施和智能体 RL 算法，以提升后训练效率和真实编码表现。 \
  <a href="https://arxiv.org/abs/2602.15763"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://z.ai/blog/glm-5"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/zai-org/GLM-5"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/zai-org/GLM-5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Kimi K2.5: Visual Agentic Intelligence** (2026.02) \
  **描述**: 该论文提出一个开源多模态智能体模型，通过统一预训练、SFT 和强化学习共同优化文本与视觉能力。它还提出 Agent Swarm 并行编排框架，用于拆解和协同执行复杂任务。 \
  <a href="https://arxiv.org/abs/2602.02276"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://www.kimi.com/ai-models/kimi-k2-5"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MoonshotAI/Kimi-K2.5"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/moonshotai/Kimi-K2.5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **MiMo-V2-Flash Technical Report** (2026.01) \
  **描述**: MiMo-V2-Flash 是一个 309B 总参数、15B 激活参数的 MoE 基础模型，通过混合滑动窗口/全局注意力、27T token 预训练和 256k 长上下文扩展，面向快速推理、代码和智能体任务。它提出 Multi-Teacher On-Policy Distillation 来扩展后训练，并将 multi-token prediction 复用为 speculative decoding 的草稿模型以提升解码速度。 \
  <a href="https://arxiv.org/abs/2601.02780"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/xiaomimimo/MiMo-V2-Flash"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models** (2026.01) \
  **描述**: 该论文提出与 MoE 条件计算互补的条件记忆稀疏轴，并以 Engram 实现静态知识的常数时间查找。论文通过缩放定律指导神经计算与记忆容量分配，使 Engram 在参数量和 FLOPs 对齐时提升知识、推理、代码、数学及长上下文检索表现。 \
  <a href="https://arxiv.org/abs/2601.07372"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/deepseek-ai/Engram"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models** (2025.12) \
  **描述**: DeepSeek-V3.2 是一个开放大语言模型，将高效长上下文计算与强推理、智能体能力结合起来。其关键技术包括 DeepSeek Sparse Attention、可扩展 RL 后训练，以及用于提升工具调用泛化和指令跟随鲁棒性的大规模智能体任务合成管线。 \
  <a href="https://arxiv.org/abs/2512.02556"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://api-docs.deepseek.com/news/news251201"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models** (2025.08) \
  **描述**: GLM-4.5 提出开源 MoE 基础模型，并通过思考/直接回答两种混合推理模式更好支持智能体、推理和代码任务。它结合大规模预训练与 RL 后训练，发布全量和紧凑版本，并在多个基准上取得强表现。 \
  <a href="https://arxiv.org/abs/2508.06471"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/zai-org/GLM-4.5"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/zai-org/GLM-4.5"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Kimi K2: Open Agentic Intelligence** (2025.07) \
  **描述**: Kimi K2 是一个万亿参数 MoE 语言模型，聚焦强智能体、推理与代码能力以及稳定的大规模训练。论文提出带 QK-clip 的 MuonClip，以提升预训练过程中的优化稳定性和 token 效率。 \
  <a href="https://arxiv.org/abs/2507.20534"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://moonshotai.github.io/Kimi-K2/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MoonshotAI/Kimi-K2"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/moonshotai/Kimi-K2-Base"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen3 Technical Report** (2025.05) \
  **描述**: 该报告介绍 Qwen3 系列，覆盖多种规模的 dense 与 MoE 模型，并强调更强的多语言表现和效率。它在同一框架中统一深思模式与快速响应模式，并通过扩展后训练提升推理、代码和智能体行为。 \
  <a href="https://arxiv.org/abs/2505.09388"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://qwen.ai/blog?id=qwen3"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen3"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen3"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **MiniMax-01: Scaling Foundation Models with Lightning Attention** (2025.01) \
  **描述**: MiniMax-01 提出基于 Lightning Attention 和 MoE 的长上下文模型家族，以提升扩展效率和实际吞吐。它结合优化后的并行策略与通信-计算重叠，在训练大模型时获得更强的长上下文表现。 \
  <a href="https://arxiv.org/abs/2501.08313"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://www.minimax.io/news/minimax-01-series-2"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-01"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeek-V3 Technical Report** (2024.12) \
  **描述**: DeepSeek-V3 是一个 671B 总参数、每 token 激活 37B 参数的 MoE 语言模型，面向高效推理和低成本大规模训练。它在 MLA 与 DeepSeekMoE 基础上引入无辅助损失的负载均衡和 multi-token prediction 训练目标，并通过稳定的 14.8T token 预训练及 SFT/RL 后训练取得强开源模型表现。 \
  <a href="https://arxiv.org/abs/2412.19437"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** (2024.09) \
  **描述**: 该论文介绍 Qwen2.5-Math 数学专用模型家族，在预训练、后训练和推理阶段持续使用自我改进。该方法增强了多种规模模型的数学推理和工具辅助解题能力。 \
  <a href="https://arxiv.org/abs/2409.12122"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2.5-Math"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen25-math-66f7162f2be749b8a8e63c8a"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen2 Technical Report** (2024.07) \
  **描述**: 该报告介绍 Qwen2 系列 dense 与 MoE 语言模型，覆盖不同规模的 base 和 instruction-tuned 版本。它强调更强的多语言、代码、数学和推理能力，并保持与闭源系统的竞争力。 \
  <a href="https://arxiv.org/abs/2407.10671"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen2-6641dc1d5fbb1a48c8708a52"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** (2024.05) \
  **描述**: DeepSeek-V2 是一个 236B 总参数、每 token 激活 21B 参数并支持 128K 上下文的 MoE 语言模型，面向低成本训练和高效推理。它结合用于压缩 KV 缓存的 Multi-head Latent Attention 与 DeepSeekMoE 稀疏计算，在降低训练成本和 KV 缓存的同时提升吞吐与开源模型表现。 \
  <a href="https://arxiv.org/abs/2405.04434"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## 推理
- **Large Language Models Explore by Latent Distilling** (2026.04) \
  **描述**: 该论文提出 Exploratory Sampling (ESamp)，用于缓解标准随机采样主要产生词面变化、语义探索不足的问题。它在测试时训练轻量 Distiller 由浅层表征预测深层表征，并用预测误差作为新颖性信号重加权候选 token，从而提升推理模型的 Pass@k 效率。 \
  <a href="https://arxiv.org/abs/2604.24927"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/LinesHogan/tLLM"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Caterpillar of Thoughts: The Optimal Test-Time Algorithm for Large Language Models** (2026.03) \
  **描述**: 该论文为 LLM 的最优测试时计算提出理论框架，证明最优算法总会生成一种“毛毛虫树”结构，并提出 CaT。该方法相比 Tree-of-Thoughts 用更少 token 生成取得更高成功率。 \
  <a href="https://arxiv.org/abs/2603.22784"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

## 检测
- **Base Models Look Human To AI Detectors** (2026.05) \
  **描述**: 该论文发现，商业 AI 文本检测器常把基础模型输出判为比指令微调模型输出更像人类文本，说明检测器可能更多捕捉指令微调痕迹和局部上下文，而非稳定的机器生成文本信号。论文提出 Humanization by Iterative Paraphrasing (HIP)，通过检测器无关的微调与迭代改写管线，在保持语义的同时提升检测规避效果。 \
  <a href="https://arxiv.org/abs/2605.19516"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

# 多模态大模型

## 视觉语言
- **Lance: Unified Multimodal Modeling by Multi-Task Synergy** (2026.05) \
  **描述**: Lance 提出一种轻量级原生统一多模态模型，在不主要依赖容量扩展的情况下支持图像和视频理解、生成与编辑。它结合共享交错上下文建模、解耦能力路径、双流 MoE、模态感知 RoPE 和分阶段多任务训练，同时提升生成与理解能力。 \
  <a href="https://arxiv.org/abs/2605.18678"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://lance-project.github.io/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/bytedance/Lance"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/bytedance-research/Lance"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding** (2026.04) \
  **描述**: 该论文提出 Video-MME-v2，一个改进的视频理解基准，用于缓解现有基准分数饱和的问题。它指出膨胀的排行榜分数往往无法真实反映模型能力，并推动更全面的视频理解评测。 \
  <a href="https://arxiv.org/abs/2604.05015"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://video-mme-v2.netlify.app/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MME-Benchmarks/Video-MME-v2"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators** (2026.04) \
  **描述**: V-Reflection 通过“先思考、再观察”的反思机制，把 MLLM 从被动视觉消费者转变为主动提问者，并让每个推理步骤都扎根于视觉证据。两阶段蒸馏设计在保持自回归高效推理的同时提升细粒度感知。 \
  <a href="https://arxiv.org/abs/2604.03307"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://idea-research.github.io/V-Reflection/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/IDEA-Research/V-Reflection"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/garlandchou/V-Reflection"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model** (2026.03) \
  **描述**: 该论文提出由 VLM 引导的 JEPA 式潜在世界模型框架，通过双时间尺度设计结合密集帧动态预测与长程语义引导。它还引入分层金字塔表征提取模块，将多层 VLM 推理特征迁移到潜在预测中，以提升手部操作轨迹预测的鲁棒性。 \
  <a href="https://arxiv.org/abs/2603.22281"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Qwen3-VL Technical Report** (2025.11) \
  **描述**: 该报告介绍 Qwen3-VL，这是 Qwen 系列迄今能力最强的视觉语言模型，在广泛多模态基准上取得更优表现。它原生支持最高 256K token 的交错上下文，可无缝融合文本、图像和视频。 \
  <a href="https://arxiv.org/abs/2511.21631"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen3-VL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen3-vl"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency** (2025.08) \
  **描述**: 该论文提出 InternVL 3.5，一个新的开源多模态模型家族，在通用性、推理能力和推理效率上显著提升。其核心包括 Cascade Reinforcement Learning 框架，用于进一步增强多模态能力。 \
  <a href="https://arxiv.org/abs/2508.18265"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Qwen2.5-VL Technical Report** (2025.02) \
  **描述**: 该技术报告介绍 Qwen2.5-VL，一个旗舰视觉语言模型，具备更强的视觉识别、精确定位、文档解析和长视频理解能力。它还通过更好的 grounding 和结构化感知提升视觉环境中的智能体交互。 \
  <a href="https://arxiv.org/abs/2502.13923"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2.5-VL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen25-vl-67ad1d2357b3028d1e9c4d56"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling** (2024.12) \
  **描述**: 该论文介绍 InternVL 2.5，一个先进的多模态 LLM 系列，也是首个在 MMMU 基准上超过 70% 的开源 MLLM。它通过 Chain-of-Thought 推理带来 3.7 个百分点提升。 \
  <a href="https://arxiv.org/abs/2412.05271"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://huggingface.co/spaces/OpenGVLab/InternVL"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** (2024.09) \
  **描述**: 该论文提出 Qwen2-VL，一个视觉语言模型系列，通过 Naive Dynamic Resolution 处理任意分辨率图像，并用 M-RoPE 融合文本、图像和视频的位置表示。论文将模型扩展到 2B、8B 和 72B 参数并扩大多模态数据，在图像、视频、多语言 OCR、文档理解和视觉智能体交互任务上取得有竞争力的表现。 \
  <a href="https://arxiv.org/abs/2409.12191"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://qwenlm.github.io/blog/qwen2-vl/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen2-VL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen2-vl"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models** (2024.07) \
  **描述**: 该论文提出 LLaVA-NeXT-Interleave，同时处理多图像、多帧视频、多视角 3D 和多 patch 场景。它将视觉指令微调扩展到更复杂的多模态输入设置。 \
  <a href="https://arxiv.org/abs/2407.07895"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/LLaVA-VL/LLaVA-NeXT"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **How Good is my Video LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs** (2024.05) \
  **描述**: 该论文提出 CVRR-ES，一个覆盖 11 类真实世界视频维度的视频 LMM 评测套件。它评估 9 个近期模型，并发现多数开源 Video-LMM 在复杂视频的鲁棒性和推理上仍然薄弱。 \
  <a href="https://arxiv.org/abs/2405.03690"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://mbzuai-oryx.github.io/CVRR-Evaluation-Suite/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond** (2023.08) \
  **描述**: 该论文提出 Qwen-VL，一个基于 Qwen-LM 构建的视觉语言模型系列，结合视觉接收器、多模态输入输出接口、三阶段训练流程和多语言多模态语料。通过对齐图像、描述和边界框三元组，Qwen-VL 获得视觉理解、定位和图中文字识别能力，并在多项视觉基准上取得强劲表现。 \
  <a href="https://arxiv.org/abs/2308.12966"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen-VL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/Qwen/Qwen-VL"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **LLaVA: Visual Instruction Tuning** (2023.04) \
  **描述**: 该论文提出 LLaVA，一个使用机器生成指令微调数据端到端训练的大型多模态模型。它展现出强多模态对话能力，并在 Science QA 上取得当时最优结果。 \
  <a href="https://arxiv.org/abs/2304.08485"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://llava-vl.github.io/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

## 多模态推理
- **Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey** (2025.03) \
  **描述**: 这篇综述弥补了多模态大模型中多模态 Chain-of-Thought 推理缺少最新系统综述的问题，覆盖图像、视频、语音、音频、3D 和结构化数据。它给出基础定义、完整 taxonomy、跨应用方法分析，并总结未来多模态推理研究的开放挑战。 \
  <a href="https://arxiv.org/abs/2503.12605"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/yaotingwangofficial/Awesome-MCoT"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

## 视觉-语言-动作
- **Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation** (2026.04) \
  **描述**: OneVL 面向 VLA 自动驾驶中的实时轨迹规划，将 Chain-of-Thought 推理压缩为由语言重建和未来帧预测共同监督的紧凑潜在 token。其三阶段训练流程让潜在推理在保持 answer-only 推理延迟的同时超过显式 CoT。 \
  <a href="https://arxiv.org/abs/2604.18486"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://xiaomi-embodied-intelligence.github.io/OneVL/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/xiaomi-research/onevl"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/xiaomi-research/onevl-models"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

# 嵌入模型
- **Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models** (2025.06) \
  **描述**: 该论文提出 Qwen3 Embedding，一个基于 Qwen3 基础模型构建的文本嵌入与重排序模型系列，结合多阶段训练、模型合并和 LLM 合成的多语言数据。该系列覆盖 0.6B、4B 和 8B 规模，并在多语言嵌入、检索、重排序、代码检索和跨语言检索基准上取得领先表现。 \
  <a href="https://arxiv.org/abs/2506.05176"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://qwenlm.github.io/blog/qwen3-embedding/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/QwenLM/Qwen3-Embedding"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/Qwen/qwen3-embedding"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

# 监督微调

## 监督微调方法
- **Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning** (2026.05) \
  **描述**: 该论文系统研究监督微调中的基于难度的数据选择，指出不存在普适最优的数据难度。论文用分布内泛化与外推之间的权衡解释数据规模相关的最优难度，并发现随着数据预算增加，最优训练样本会逐渐转向更难样本。 \
  <a href="https://arxiv.org/abs/2605.12906"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability** (2026.04) \
  **描述**: 该论文挑战“SFT 只会记忆而 RL 才能泛化”的常见说法，发现带长链式思维监督的推理 SFT 也能跨域泛化。其泛化效果取决于优化动态、训练数据和基础模型能力三者的共同作用。 \
  <a href="https://arxiv.org/abs/2604.06628"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/Nebularaid2000/rethink_sft_generalization"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/jasonrqh/rethink-sft-generalization"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection** (2026.01) \
  **描述**: 该论文提出 ProFit，一种监督微调方法，通过 token 概率作为语义重要性代理并屏蔽低概率 token，缓解单参考答案过拟合。该方法让训练更聚焦核心逻辑内容，并在推理和数学任务上优于标准 SFT 基线。 \
  <a href="https://arxiv.org/abs/2601.09195"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

# 训练

## 优化
- **PowLU: An Activation Function for Stable Pre-Training of LLMs** (2026.05) \
  **描述**: 该论文指出 SwiGLU 在大正输入下接近二次放大会扩大输出范围并加剧 outlier，从而在低精度大规模 LLM 预训练中带来数值不稳定。论文提出 Power Linear Unit (PowLU)，用有理幂函数在保持自适应非线性的同时稳定 spike 区域，并通过 scaling law 与 Ling 模型实验展示竞争性效果和更好的训练可扩展性。 \
  <a href="https://arxiv.org/abs/2605.25704"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

## 蒸馏
- **Knowledge Distillation of Large Language Models** (2023.06) \
  **描述**: 该论文研究面向生成式大语言模型的白盒知识蒸馏，并提出 MiniLLM，用反向 KLD 替代标准前向 KLD，以避免学生模型高估教师分布中的低概率区域。论文推导了有效优化方法，并在 120M 到 13B 参数的不同模型族上提升指令跟随质量、校准性、曝光偏差和长文本生成表现。 \
  <a href="https://arxiv.org/abs/2306.08543"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/microsoft/LMOps/tree/main/minillm"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/MiniLLM"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

# 强化学习

## 策略优化
- **Rethinking the Divergence Regularization in LLM RL** (2026.06) \
  **描述**: 该论文提出 Divergence Regularized Policy Optimization (DRPO)，以平滑的优势加权二次正则项替代 DPPO 的硬散度掩码，同时保留其信赖域几何结构。DRPO 通过有界连续梯度权重和越界后的纠正信号，提高 LLM 强化学习训练的稳定性与效率。 \
  <a href="https://arxiv.org/abs/2606.09821"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/Tencent-Hunyuan/UniRL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Constraint-Infused Policy Optimization: Principles and Practices for Harnessing Advanced LLM Reasoning** (2026.05) \
  **描述**: 该论文将大语言模型强化学习表述为约束策略优化，通过不同约束选择统一现有算法，并揭示裁剪、KL 正则化与信赖域的作用。论文据此推导 Constraint-Infused Policy Optimization (CIPO)，在多种任务和模型族上提升推理性能与训练稳定性。 \
  <a href="https://arxiv.org/abs/2605.16826"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/trestad/CIPO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Group Sequence Policy Optimization** (2025.07) \
  **描述**: 该论文提出 GSPO，一种用于大语言模型的强化学习算法，以序列级似然比替代 token 级重要性比率，并在序列级执行裁剪、奖励与优化。GSPO 相比 GRPO 提升训练效率和性能，稳定 MoE 强化学习训练，并有助于简化 Qwen3 模型的大规模强化学习基础设施。 \
  <a href="https://arxiv.org/abs/2507.18071"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://qwen.ai/blog?id=gspo"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

- **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (2025.03) \
  **描述**: 该论文提出 Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)，一个用于激发大语言模型推理能力的开源大规模强化学习系统。论文开放训练配方、代码、数据集和模型权重，基于 Qwen2.5-32B 在 AIME 2024 上达到 50 分，提升大规模 LLM 强化学习的可复现性。 \
  <a href="https://arxiv.org/abs/2503.14476"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://dapo-sia.github.io/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/BytedTsinghua-SIA/DAPO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **KTO: Model Alignment as Prospect Theoretic Optimization** (2024.02) \
  **描述**: 该论文将成功的大语言模型对齐损失归纳为带有 prospect theory 偏置的 human-aware losses，并提出 KTO，直接利用 desirable/undesirable 二元反馈优化生成效用。KTO 在 1B 到 30B 规模上达到或超过基于成对偏好的方法，也说明最佳对齐损失取决于具体场景中的归纳偏置。 \
  <a href="https://arxiv.org/abs/2402.01306"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (2023.05) \
  **描述**: 该论文提出 Direct Preference Optimization (DPO)，重新参数化 RLHF 奖励模型，使最优策略能够通过简单的分类损失直接从偏好数据中学习。DPO 无需单独拟合奖励模型或执行在线强化学习，以更简单稳定的训练达到或超过基于 PPO 的 RLHF。 \
  <a href="https://arxiv.org/abs/2305.18290"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/eric-mitchell/direct-preference-optimization"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Proximal Policy Optimization Algorithms** (2017.07) \
  **描述**: 该论文提出 Proximal Policy Optimization (PPO)，一类在环境采样与代理目标上的多轮小批量优化之间交替进行的策略梯度方法。PPO 保留信赖域方法的关键优势，同时更易实现，并在样本效率、性能与运行时间之间取得良好平衡。 \
  <a href="https://arxiv.org/abs/1707.06347"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://openai.com/index/openai-baselines-ppo/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/openai/baselines"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## OPD
- **PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation** (2026.06) \
  **描述**: 该论文指出，基于采样 token 的 on-policy distillation 中，无界的 log-ratio 奖励会引发高方差梯度与训练不稳定。PowerOPD 使用由 Box-Cox 幂变换导出的有界、符号一致奖励族，在降低全词表 OPD 时间和显存开销的同时提升数学推理准确率。 \
  <a href="https://arxiv.org/abs/2606.17199"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information** (2026.05) \
  **描述**: 该论文分析 on-policy self-distillation 在数学推理中效果不稳定的原因，指出基于特权上下文的教师信号会通过点式互信息过度强化答案已暗示的 token，同时压低驱动多步搜索的思考 token。论文提出 AntiSD，用带熵门控的反向自蒸馏散度替代默认下降方向，在 2 到 10 倍更少训练步数内达到 GRPO 水平，并将最终准确率最高提升 11.5 个百分点。 \
  <a href="https://arxiv.org/abs/2605.11609"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/FloyedShen/AntiSD"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Draft-OPD: Adapting Speculative Draft Models from LLMs via On-Policy Distillation** (2026.05) \
  **描述**: 该论文提出 Draft-OPD，通过 on-policy distillation 复用 RL 训练轨迹，将 RL 后训练 LLM 的能力适配到 speculative draft model，避免为 draft model 进行昂贵的在线生成。论文证明 RL 训练与 OPD 式蒸馏之间的等价关系，并在保持任务性能的同时将 speculative decoding 速度最高提升 2.14 倍。 \
  <a href="https://arxiv.org/abs/2605.29343"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://www.haodilei.top/draft-opd/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/bingyang-lei/Draft-OPD"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/bingyang-lei/draft-opd"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification** (2026.05) \
  **描述**: 该论文提出 OmniOPD，一种无 logits 的在策略蒸馏框架，以基于语义相似度评分的 Monte Carlo 分块 rollout 替代脆弱的 token 级 logit matching，从而支持黑盒教师。其 peak-entropy 调度器聚焦高不确定性推理分叉，并结合贝叶斯平滑与基础模型 KL 锚点稳定训练，在数学任务上相较标准 OPD 最高提升 28.64%。 \
  <a href="https://arxiv.org/abs/2606.01476"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Self-Distilled RLVR** (2026.04) \
  **描述**: 该论文研究 RLVR 中的 on-policy self-distillation，指出仅依赖带特权信息的自教师信号会造成信息泄漏和长期训练不稳定。论文提出 RLSD，用自蒸馏估计 token 级更新幅度，同时保留 RLVR 的环境反馈作为可靠更新方向。 \
  <a href="https://arxiv.org/abs/2604.03128"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** (2026.02) \
  **描述**: 该论文证明 on-policy distillation 是 dense KL-constrained RL 的一个特例，并提出带灵活参考模型和奖励缩放因子的 G-OPD。其 reward extrapolation 版本 ExOPD 相比标准 OPD 更强，并能在融合 RL 训练的领域专家时让学生模型超越教师边界。 \
  <a href="https://arxiv.org/abs/2602.12125"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/RUCBM/G-OPD"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/Keven16/G-OPD-Training-Data"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models** (2026.01) \
  **描述**: 该论文提出 On-Policy Self-Distillation (OPSD)，让同一 LLM 在不同上下文下同时扮演教师和学生：教师可看到带特权的验证推理轨迹，学生只看到问题并从自身策略采样。通过在学生的 on-policy rollout 上匹配 token 级分布，OPSD 无需外部教师即可提供密集监督，并在数学推理上比 GRPO 和 off-policy 蒸馏更具 token 效率。 \
  <a href="https://arxiv.org/abs/2601.18734"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/siyan-zhao/OPSD"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Self-Distillation Enables Continual Learning** (2026.01) \
  **描述**: 该论文提出 Self-Distillation Fine-Tuning (SDFT)，一种从专家示范中进行 on-policy self-distillation 的方法，通过让示范条件下的模型作为自身教师生成训练信号。该方法在学习新任务的同时显著减少灾难性遗忘，使单个模型无需显式奖励函数也能顺序积累技能与知识。 \
  <a href="https://arxiv.org/abs/2601.19897"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://self-distillation.github.io/SDFT"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/idanshen/Self-Distillation"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## 奖励建模
- **MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels** (2026.03) \
  **描述**: 该论文提出 MemReward，一个基于图经验记忆的奖励预测框架，在有限标签下让 3B 和 1.5B 模型分别达到 Oracle 表现的 97.3% 和 96.6%。它还在域外任务上超过 Oracle。 \
  <a href="https://arxiv.org/abs/2603.19310"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/ulab-uiuc/MemReward"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/ulab-ai/MemReward"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Scaling Reward Modeling without Human Supervision** (2026.03) \
  **描述**: 该论文研究无需人工标注的无监督奖励模型扩展，通过学习网页语料文档前缀与后缀之间的偏好来训练奖励模型。实验显示该方法在不同模型骨干上稳定提升 RewardBench，并改进 best-of-N 选择和策略优化。 \
  <a href="https://arxiv.org/abs/2603.02225"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Reward Modeling from Natural Language Human Feedback** (2026.01) \
  **描述**: 该论文将偏好数据上的 RLVR 用于训练生成式奖励模型，指出二分类任务会让 GRM 倾向于猜对结果而非给出可靠批判。论文提出方法缓解这一问题。 \
  <a href="https://arxiv.org/abs/2601.07349"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models** (2025.10) \
  **描述**: 这篇综述梳理过程奖励模型如何在步骤或轨迹层面评估并引导大语言模型推理，而不只判断最终答案。论文围绕过程数据生成、PRM 构建，以及 PRM 在测试时扩展和强化学习中的使用闭环，覆盖数学、代码、多模态推理、机器人和智能体等应用。 \
  <a href="https://arxiv.org/abs/2510.08049"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

## 视频生成强化学习
- **KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration** (2026.05) \
  **描述**: KVPO 使用 ODE-native online GRPO 框架，将流式自回归视频生成器与人类偏好对齐。它用历史 KV 缓存的因果语义路由替代噪声探索，并基于 Trajectory Velocity Energy 优化速度场代理策略。 \
  <a href="https://arxiv.org/abs/2605.14278"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://richard-zhang-ai.github.io/KVPO-Project/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/Richard-Zhang-AI/KVPO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/Richard-ZZZZZ/KVPO"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## 多模态强化学习
- **CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning** (2025.09) \
  **描述**: 该论文提出 CapRL，首次将 RLVR 应用于开放式图像描述，通过检验无视觉语言模型能否仅依据生成描述回答图像问题来构造奖励。训练得到的 CapRL-3B 能生成信息更丰富且更多样的描述，其生成的描述数据还在 12 个基准上提升了大视觉语言模型的预训练效果。 \
  <a href="https://arxiv.org/abs/2509.22647"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/InternLM/CapRL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/internlm/CapRL-3B"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## 推理强化学习
- **All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models** (2026.04) \
  **描述**: 该论文提出 MUPO，一种强化学习方法，通过激励多解之间的发散思考来缓解 GRPO 训练 VLM 时的多样性坍塌。它让模型形成更深且更广的推理模式。 \
  <a href="https://arxiv.org/abs/2604.00479"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://xytian1008.github.io/MUPO/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/xytian1008/MUPO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/xytian1008/MUPO-Thinker-7B"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning** (2026.04) \
  **描述**: 该论文提出 VL-Calibration，一种将大视觉语言模型的视觉置信度与推理置信度解耦的强化学习框架，用于缓解模型高置信度错误预测。它结合图像扰动下的视觉定位与 token 熵估计视觉确定性，并通过 token 级优势重加权提升校准效果和视觉推理准确率。 \
  <a href="https://arxiv.org/abs/2604.09529"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/Mr-Loevan/VL-Calibration"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **The Art of Efficient Reasoning: Data, Reward, and Optimization** (2026.03) \
  **描述**: 该论文研究 LLM 的高效推理，通过 RL 激励短而准确的推理轨迹。论文总结了训练阶段、奖励设计和优化策略，并在 0.6B 到 30B 模型上分析泛化规律。 \
  <a href="https://arxiv.org/abs/2602.20945"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://wutaiqiang.github.io/project/Art"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

- **FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization** (2026.03) \
  **描述**: 该论文提出 FIPO，一种强化学习算法，用于解决 LLM 推理瓶颈中的粗粒度信用分配问题。它针对 GRPO 式训练中结果奖励无法区分关键逻辑转折与普通 token 的问题进行改进。 \
  <a href="https://arxiv.org/abs/2603.19835"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis** (2026.02) \
  **描述**: 该论文提出 Agentic Proposing，一个使用专门智能体和 Multi-Granularity Policy Optimization 动态选择、组合模块化推理技能的框架。该框架用于合成高精度训练轨迹，从而增强大语言模型推理能力。 \
  <a href="https://arxiv.org/abs/2602.03279"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** (2025.05) \
  **描述**: 该论文研究推理语言模型强化学习中的策略熵坍塌瓶颈，发现熵与下游性能之间存在经验关系，使性能上限可被预测。论文从动作概率与 logit 更新的协方差推导熵动态，并提出 Clip-Cov 和 KL-Cov 保持探索、提升下游表现。 \
  <a href="https://arxiv.org/abs/2505.22617"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025.01) \
  **描述**: 该论文展示纯强化学习无需人工标注推理轨迹即可直接激发 LLM 的高级推理行为。所提出框架诱导自我反思、验证和自适应策略使用，并在数学、代码和 STEM 推理任务上取得强提升。 \
  <a href="https://arxiv.org/abs/2501.12948"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024.02) \
  **描述**: 该论文提出 DeepSeekMath 7B，将精心设计的网页级数学数据筛选流程与 Group Relative Policy Optimization (GRPO) 相结合；GRPO 是 PPO 的一种变体。该方法在降低 PPO 显存占用的同时增强数学推理能力，无需外部工具或投票即可在竞赛级 MATH 基准上取得强劲表现。 \
  <a href="https://arxiv.org/abs/2402.03300"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/deepseek-ai/deepseek-math"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/deepseek-ai/deepseek-math-7b-rl"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

## 智能体强化学习
- **Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning** (2026.05) \
  **描述**: Spreadsheet-RL 是一个用于在真实 Microsoft Excel 环境中训练专用表格智能体的强化学习微调框架，面向提示式智能体难以处理的复杂多步工作流。它结合自动化起止表格数据构建、多轮 Spreadsheet Gym 沙盒工具环境，以及 Domain-Spreadsheet 基准，以提升真实表格自动化能力。 \
  <a href="https://arxiv.org/abs/2605.22642"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://spreadsheet-rl.github.io/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/Spreadsheet-RL/Spreadsheet-RL"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/Spreadsheet-RL/Spreadsheet-RL"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration** (2026.02) \
  **描述**: 该论文提出 Actor-Refiner 协作机制，解决搜索增强推理 RL 中的多尺度信用分配问题。它缓解稀疏轨迹级奖励难以区分高质量推理与偶然猜对的问题，并减少冗余或误导性搜索行为。 \
  <a href="https://arxiv.org/abs/2602.03647"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Arena-RL: Training LLMs as Game Players with Vision-Language Action Models** (2026.01) \
  **描述**: 该论文提出 Arena-RL，一个通过视觉语言动作模型训练 LLM 驱动智能体玩视觉游戏的强化学习框架，重点从交互式游戏反馈中改进策略。实验表明，基于游戏轨迹的奖励优化能显著提升策略决策与跨游戏泛化。 \
  <a href="https://arxiv.org/abs/2601.06487"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Agentic Reinforced Policy Optimization** (2025.07) \
  **描述**: 该论文提出 ARPO，一种面向多轮 LLM 智能体训练的智能体强化学习算法，用于平衡长程推理能力与逐步工具交互能力。它通过基于熵的自适应 rollout 采样和工具调用步骤上的优势归因，在计算推理、知识推理和深度搜索基准上优于轨迹级 RL 方法，并将工具调用预算约减半。 \
  <a href="https://arxiv.org/abs/2507.19849"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/RUC-NLPIR/ARPO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025.03) \
  **描述**: 该论文提出 Search-R1，一个让 LLM 在逐步推理中通过 RL 学会自主生成搜索查询并使用实时检索的框架。它提升了模型获取外部知识和最新信息的能力。 \
  <a href="https://arxiv.org/abs/2503.09516"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/PeterGriffinJin/Search-R1"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Search-o1: Agentic Search-Enhanced Large Reasoning Models** (2025.01) \
  **描述**: 该论文提出 Search-o1，一个用智能体式检索增强生成机制和 Reason-in-Documents 模块增强大推理模型的框架。它通过精炼检索文档来缓解长程推理中的知识不足。 \
  <a href="https://arxiv.org/abs/2501.05366"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/sunnynexus/Search-o1"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## 视觉-语言-动作强化学习
- **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models** (2025.11) \
  **描述**: 该论文提出 SRPO，一个用于视觉语言动作模型的强化学习框架，用模型自身成功轨迹中的进展式奖励替代稀疏二元奖励。它利用潜在世界模型表征稳健衡量行为进展，并以更少 RL 步数在 LIBERO 上取得最优操作成功率。 \
  <a href="https://arxiv.org/abs/2511.15605"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

# 智能体应用

## 工具调用
- **Is Grep All You Need? How Agent Harnesses Reshape Agentic Search** (2026.05) \
  **描述**: 该论文实证研究检索策略、智能体 harness 设计与工具结果呈现方式在 agentic search 中如何相互影响。在基于 LongMemEval、Chronos 和厂商 CLI harness 的实验中，grep 往往优于向量检索，但整体性能仍强烈受 harness 与工具调用风格影响。 \
  <a href="https://arxiv.org/abs/2605.15184"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Thinking with Programming Vision: Towards a Unified View for Thinking with Images** (2025.12) \
  **描述**: 该论文指出当前多模态工具调用推理在简单图像旋转和损坏下仍然脆弱，并提出 CodeVision，一种让模型通过生成代码调用任意图像操作的 code-as-tool 框架。它结合 SFT、RL 和密集过程奖励，提升多工具推理、执行效率和错误恢复。 \
  <a href="https://arxiv.org/abs/2512.03746"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/ByteDance-BandAI/CodeVision"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## AI 研究
- **AI for Auto-Research: Roadmap & User Guide** (2026.05) \
  **描述**: 这篇综述从创造、写作、验证和传播等环节分析 AI 辅助科研，指出自动化在哪些地方可靠，以及自治系统在新颖性、实验和科学判断上仍会失败。它提供生命周期分类、基准套件、工具清单、设计原则和面向实践者的人类治理式 AI 研究工作流指南。 \
  <a href="https://arxiv.org/abs/2605.18661"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://worldbench.github.io/awesome-ai-auto-research"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/worldbench/awesome-ai-auto-research"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs** (2026.05) \
  **描述**: 该论文提出 Crafter，一个面向多种图类型和输入条件的多智能体科学图生成 harness，并提出 CraftEditor 将栅格输出转换为可编辑 SVG。论文还构建带人工质量标注的 CraftBench 基准，并展示其相较独立生成器和 agentic baseline 的优势。 \
  <a href="https://arxiv.org/abs/2605.30611"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/HaozheZhao/Crafter"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/BleachNick/CraftBench"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **AIRA_2: Overcoming Bottlenecks in AI Research Agents** (2026.03) \
  **描述**: 该论文提出 AIRA_2，一种 AI 研究智能体架构，用于解决实验吞吐有限、基于噪声验证的选择不稳定以及单轮静态算子等瓶颈。它结合异步多 GPU worker、Hidden Consistent Evaluation 和交互式 ReAct 智能体，提升长程研究任务表现。 \
  <a href="https://arxiv.org/abs/2603.26499"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

## 智能体技能
- **SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution** (2026.05) \
  **描述**: SkillsVote 将 Agent Skills 视为可复用经验资产，通过收集、推荐、归因和演化治理来缓解智能体轨迹噪声与难治理问题。它分析大规模开源技能语料，在执行前推荐结构化技能上下文，并只接纳证据门控的成功发现，以在不更新模型的情况下改进冻结智能体。 \
  <a href="https://arxiv.org/abs/2605.18401"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://skills.vote/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>
  <a href="https://github.com/MemTensor/skills-vote"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

- **From Context to Skills: Can Language Models Learn from Context Skillfully?** (2026.04) \
  **描述**: Ctx2Skill 面向长且密集上下文中的 context learning，解决人工标注技能成本高、自动构建技能缺少外部反馈的问题。它通过多智能体 self-play 与 Cross-time Replay 自动发现、细化并选择可复用自然语言技能，从而提升不同语言模型的上下文学习解题率。 \
  <a href="https://arxiv.org/abs/2604.27660"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/S1s-Z/Ctx2Skill"><img src="assets/icons/github.svg" alt="代码" width="20"></a>
  <a href="https://huggingface.co/datasets/ssz1111/Ctx2Skill"><img src="assets/icons/huggingface.svg" alt="Hugging Face" width="20"></a>

- **SkillReducer: Optimizing LLM Agent Skills for Token Efficiency** (2026.03) \
  **描述**: 该论文提出 SkillReducer，一个两阶段优化框架，用于压缩 LLM 智能体技能这类预封装指令集。它在提升功能质量 2.8% 的同时，将技能描述和正文分别压缩 48% 和 39%，降低 token 成本与注意力稀释。 \
  <a href="https://arxiv.org/abs/2603.29919"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

## 智能体开发
- **Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization** (2026.03) \
  **描述**: 该论文提出 Nurture-First Development，一种通过结构化对话而非固定代码优先或提示优先方式培养领域专家智能体的范式。它形式化了知识结晶循环、三层认知架构、双工作区模式和螺旋开发模型，用于持续把从业者隐性知识转化为可复用智能体资产。 \
  <a href="https://arxiv.org/abs/2603.10808"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Controlled Self-Evolution for Algorithmic Code Optimization** (2026.01) \
  **描述**: 该论文提出 EvoControl，一个面向算法代码优化的受控自进化框架，在 generate-verify-refine 循环中平衡正确性与探索。它结合分阶段自进化、类遗传种群搜索和进化记忆，在高难算法基准上提升代码质量。 \
  <a href="https://arxiv.org/abs/2601.07348"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/QuantaAlpha/EvoControl"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## 记忆
- **Trajectory-Informed Memory Generation for Self-Improving Agent Systems** (2026.03) \
  **描述**: 该论文提出一个从 LLM 智能体执行轨迹中提取可行动经验，并在未来任务中作为上下文记忆检索的框架。它结合轨迹智能提取、决策归因、上下文学习生成和自适应记忆检索，尤其提升复杂 AppWorld 场景的任务完成率。 \
  <a href="https://arxiv.org/abs/2603.10600"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (2025.10) \
  **描述**: 该论文提出 BEAM，一个由长而连贯对话和探测问题组成的基准，用于评估 LLM 长期记忆；同时提出 LIGHT，一个包含情景记忆、工作记忆和草稿板的记忆框架。二者共同揭示长上下文记忆限制，并提升长程对话推理表现。 \
  <a href="https://arxiv.org/abs/2510.27246"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **MLP Memory: A Retriever-Pretrained Memory for Large Language Models** (2025.08) \
  **描述**: 该论文提出 MLP Memory，一个轻量级参数模块，通过预训练 MLP 模仿 kNN 检索器行为来内化检索模式。该方法在 RAG 与微调之间架起桥梁。 \
  <a href="https://arxiv.org/abs/2508.01832"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** (2025.07) \
  **描述**: 该论文提出 MemAgent，一个基于多轮对话 RL 的记忆智能体，用线性复杂度处理无限长文档。它旨在解决外推到超长上下文时性能退化的问题。 \
  <a href="https://arxiv.org/abs/2507.02259"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://memagent-sialab.github.io/"><img src="assets/icons/project.svg" alt="项目" width="20"></a>

# 视觉

## 目标检测
- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** (2022.03) \
  **描述**: DINO 通过对比去噪训练、用于锚点初始化的混合 query 选择，以及 look-forward-twice 框预测方案改进 DETR 类目标检测器。它在显著降低模型和数据需求的同时，在 COCO 上取得当时最优结果。 \
  <a href="https://arxiv.org/abs/2203.03605"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/IDEA-Research/DINO"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

# 自动提示

## 提示优化
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025.07) \
  **描述**: GEPA 提出一种提示优化器，通过自然语言反思从试错中学习高层规则，平均超过 GRPO 6%，并最多减少 35 倍 rollout。它还比 MIPROv2 高出 10% 以上，并在代码优化的推理时搜索中展现潜力。 \
  <a href="https://arxiv.org/abs/2507.19457"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>
  <a href="https://github.com/gepa-ai/gepa"><img src="assets/icons/github.svg" alt="代码" width="20"></a>

## 评测器提示
- **Becoming Experienced Judges: Selective Test-Time Learning for Evaluators** (2025.12) \
  **描述**: 该论文提出 Learning While Evaluating，让 LLM-as-a-judge 系统在推理时通过自生成反馈更新元提示，从而按序列持续改进。它进一步提出 Selective LWE，只在自我不一致样本上更新，以更好的成本效率提升评测质量。 \
  <a href="https://arxiv.org/abs/2512.06751"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **描述**: APE 通过从失败案例中自动发现辅助评测维度，并结合置信度感知选择进行集成，提升 LLM-as-a-judge 的可靠性。它更有效地使用测试时计算，从而提高与人类对齐基准的一致性。 \
  <a href="https://arxiv.org/abs/2510.06538"><img src="assets/icons/arxiv.svg" alt="论文" width="20"></a>

# 笔记

## 论文解读

- **CapRL：用强化学习激发视觉语言模型的描述能力** (2026-06-15) \
  **描述**: CapRL 用 vision-free LLM 的 MCQ 答题准确率评价 caption，将主观的描述质量评分改造成可验证奖励，并据此训练图像描述模型。 \
  [[笔记](notes/zh/reinforcement-learning/CapRL.qmd)]
  [[English](notes/en/reinforcement-learning/CapRL.qmd)]

- **从 Qwen-VL 到 Qwen3-VL：四代模型的架构与训练演进** (2026-06-15) \
  **描述**: 梳理 Qwen-VL 四代模型在视觉语言对齐、动态分辨率、时空位置编码、视频建模与深层视觉融合上的技术演进。 \
  [[笔记](notes/zh/mllms/From-Qwen-VL-to-Qwen3-VL.qmd)]
  [[English](notes/en/mllms/From-Qwen-VL-to-Qwen3-VL.qmd)]

- **Entropy Collapse：大模型 RL 训练中的策略熵消耗机制** (2026-06-18) \
  **描述**: 从策略熵、SFT 与 RL 的优化差异、DAPO 的 Clip-Higher，以及协方差正则化角度理解大模型 RL 训练中的 Entropy Collapse。 \
  [[笔记](notes/zh/reinforcement-learning/Entropy_Collapse.qmd)]
  [[English](notes/en/reinforcement-learning/Entropy_Collapse.qmd)]

## 技术思考

- **OPD：后训练中的能力整合接口** (2026-05-28) \
  **描述**: 从 Qwen3、GLM-5、MiMo-V2 与 DeepSeek-V4 的技术路径看 OPD 如何成为后训练中的能力整合接口。 \
  [[笔记](notes/zh/opd/post-training-opd.qmd)]
  [[English](notes/en/opd/post-training-opd.qmd)]

- **PPO、DPO 与 GRPO：大模型对齐算法的目标函数与训练流程** (2026-06-16) \
  **描述**: 从目标函数、优势估计和训练循环出发，对比 PPO、DPO 与 GRPO 在大模型对齐中的设计取舍与适用边界。 \
  [[笔记](notes/zh/reinforcement-learning/PPO-DPO-GRPO.qmd)]
  [[English](notes/en/reinforcement-learning/PPO-DPO-GRPO.qmd)]

# 博客

- **When AI builds itself** (2026-06-04) \
  **描述**: Anthropic Institute 结合公开基准和 Anthropic 内部数据，讨论 AI 已在加速 AI 研发，并分析递归自我改进的可能路径、风险以及提前建设监督与协调机制的必要性。 \
  [[博客](https://www.anthropic.com/institute/recursive-self-improvement)]
