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
- [强化学习](#强化学习)
  - [OPD](#opd)
  - [奖励建模](#奖励建模)
  - [视频生成强化学习](#视频生成强化学习)
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

# 注意力机制

## 注意力架构
- **Attention Residuals** (2026.03) \
  **描述**: 该工作用对前序层输出的注意力替代固定残差累积，使模型能够根据输入动态聚合不同深度的信息，并缓解 PreNorm 带来的表征稀释问题。论文还提出 Block AttnRes，在更低显存与通信开销下支持可扩展训练。 \
  [[论文](https://arxiv.org/abs/2603.15031)]
  [[项目](https://github.com/MoonshotAI/Attention-Residuals)]

# 大语言模型

## 基础模型
- **The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook** (2026.04) \
  **描述**: 这篇综述认为连续潜在空间正在成为语言模型的原生计算基底，可缓解显式 token 生成中的冗余、离散化瓶颈和语义损失。论文从机制和能力两个视角梳理该领域，并总结未来关键挑战。 \
  [[论文](https://arxiv.org/abs/2604.02029)]
  [[项目](https://github.com/YU-deep/Awesome-Latent-Space)]

- **GLM-5: from Vibe Coding to Agentic Engineering** (2026.02) \
  **描述**: GLM-5 是面向长程智能体工程的新一代基础模型，在降低训练与推理成本的同时保持长上下文能力。它引入异步 RL 基础设施和智能体 RL 算法，以提升后训练效率和真实编码表现。 \
  [[论文](https://arxiv.org/abs/2602.15763)]
  [[项目](https://z.ai/blog/glm-5)]
  [[代码](https://github.com/zai-org/GLM-5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-5)]

- **Kimi K2.5: Visual Agentic Intelligence** (2026.02) \
  **描述**: 该论文提出一个开源多模态智能体模型，通过统一预训练、SFT 和强化学习共同优化文本与视觉能力。它还提出 Agent Swarm 并行编排框架，用于拆解和协同执行复杂任务。 \
  [[论文](https://arxiv.org/abs/2602.02276)]
  [[项目](https://www.kimi.com/ai-models/kimi-k2-5)]
  [[代码](https://github.com/MoonshotAI/Kimi-K2.5)]
  [[Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.5)]

- **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models** (2025.08) \
  **描述**: GLM-4.5 提出开源 MoE 基础模型，并通过思考/直接回答两种混合推理模式更好支持智能体、推理和代码任务。它结合大规模预训练与 RL 后训练，发布全量和紧凑版本，并在多个基准上取得强表现。 \
  [[论文](https://arxiv.org/abs/2508.06471)]
  [[代码](https://github.com/zai-org/GLM-4.5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-4.5)]

- **Kimi K2: Open Agentic Intelligence** (2025.07) \
  **描述**: Kimi K2 是一个万亿参数 MoE 语言模型，聚焦强智能体、推理与代码能力以及稳定的大规模训练。论文提出带 QK-clip 的 MuonClip，以提升预训练过程中的优化稳定性和 token 效率。 \
  [[论文](https://arxiv.org/abs/2507.20534)]
  [[项目](https://moonshotai.github.io/Kimi-K2/)]
  [[代码](https://github.com/MoonshotAI/Kimi-K2)]
  [[Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Base)]

- **Qwen3 Technical Report** (2025.05) \
  **描述**: 该报告介绍 Qwen3 系列，覆盖多种规模的 dense 与 MoE 模型，并强调更强的多语言表现和效率。它在同一框架中统一深思模式与快速响应模式，并通过扩展后训练提升推理、代码和智能体行为。 \
  [[论文](https://arxiv.org/abs/2505.09388)]
  [[项目](https://qwen.ai/blog?id=qwen3)]
  [[代码](https://github.com/QwenLM/Qwen3)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen3)]

- **MiniMax-01: Scaling Foundation Models with Lightning Attention** (2025.01) \
  **描述**: MiniMax-01 提出基于 Lightning Attention 和 MoE 的长上下文模型家族，以提升扩展效率和实际吞吐。它结合优化后的并行策略与通信-计算重叠，在训练大模型时获得更强的长上下文表现。 \
  [[论文](https://arxiv.org/abs/2501.08313)]
  [[项目](https://www.minimax.io/news/minimax-01-series-2)]
  [[代码](https://github.com/MiniMax-AI/MiniMax-01)]
  [[Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)]

- **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** (2024.09) \
  **描述**: 该论文介绍 Qwen2.5-Math 数学专用模型家族，在预训练、后训练和推理阶段持续使用自我改进。该方法增强了多种规模模型的数学推理和工具辅助解题能力。 \
  [[论文](https://arxiv.org/abs/2409.12122)]
  [[代码](https://github.com/QwenLM/Qwen2.5-Math)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen25-math-66f7162f2be749b8a8e63c8a)]

- **Qwen2 Technical Report** (2024.07) \
  **描述**: 该报告介绍 Qwen2 系列 dense 与 MoE 语言模型，覆盖不同规模的 base 和 instruction-tuned 版本。它强调更强的多语言、代码、数学和推理能力，并保持与闭源系统的竞争力。 \
  [[论文](https://arxiv.org/abs/2407.10671)]
  [[代码](https://github.com/QwenLM/Qwen2)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6641dc1d5fbb1a48c8708a52)]

## 推理
- **Large Language Models Explore by Latent Distilling** (2026.04) \
  **描述**: 该论文提出 Exploratory Sampling (ESamp)，用于缓解标准随机采样主要产生词面变化、语义探索不足的问题。它在测试时训练轻量 Distiller 由浅层表征预测深层表征，并用预测误差作为新颖性信号重加权候选 token，从而提升推理模型的 Pass@k 效率。 \
  [[论文](https://arxiv.org/abs/2604.24927)]
  [[代码](https://github.com/LinesHogan/tLLM)]

- **Caterpillar of Thoughts: The Optimal Test-Time Algorithm for Large Language Models** (2026.03) \
  **描述**: 该论文为 LLM 的最优测试时计算提出理论框架，证明最优算法总会生成一种“毛毛虫树”结构，并提出 CaT。该方法相比 Tree-of-Thoughts 用更少 token 生成取得更高成功率。 \
  [[论文](https://arxiv.org/abs/2603.22784)]

## 检测
- **Base Models Look Human To AI Detectors** (2026.05) \
  **描述**: 该论文发现，商业 AI 文本检测器常把基础模型输出判为比指令微调模型输出更像人类文本，说明检测器可能更多捕捉指令微调痕迹和局部上下文，而非稳定的机器生成文本信号。论文提出 Humanization by Iterative Paraphrasing (HIP)，通过检测器无关的微调与迭代改写管线，在保持语义的同时提升检测规避效果。 \
  [[论文](https://arxiv.org/abs/2605.19516)]
  [[代码](https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing)]

# 多模态大模型

## 视觉语言
- **Lance: Unified Multimodal Modeling by Multi-Task Synergy** (2026.05) \
  **描述**: Lance 提出一种轻量级原生统一多模态模型，在不主要依赖容量扩展的情况下支持图像和视频理解、生成与编辑。它结合共享交错上下文建模、解耦能力路径、双流 MoE、模态感知 RoPE 和分阶段多任务训练，同时提升生成与理解能力。 \
  [[论文](https://arxiv.org/abs/2605.18678)]
  [[项目](https://lance-project.github.io/)]
  [[代码](https://github.com/bytedance/Lance)]
  [[Hugging Face](https://huggingface.co/bytedance-research/Lance)]

- **Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding** (2026.04) \
  **描述**: 该论文提出 Video-MME-v2，一个改进的视频理解基准，用于缓解现有基准分数饱和的问题。它指出膨胀的排行榜分数往往无法真实反映模型能力，并推动更全面的视频理解评测。 \
  [[论文](https://arxiv.org/abs/2604.05015)]
  [[项目](https://video-mme-v2.netlify.app/)]
  [[代码](https://github.com/MME-Benchmarks/Video-MME-v2)]
  [[Hugging Face](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2)]

- **V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators** (2026.04) \
  **描述**: V-Reflection 通过“先思考、再观察”的反思机制，把 MLLM 从被动视觉消费者转变为主动提问者，并让每个推理步骤都扎根于视觉证据。两阶段蒸馏设计在保持自回归高效推理的同时提升细粒度感知。 \
  [[论文](https://arxiv.org/abs/2604.03307)]
  [[项目](https://idea-research.github.io/V-Reflection/)]
  [[代码](https://github.com/IDEA-Research/V-Reflection)]
  [[Hugging Face](https://huggingface.co/garlandchou/V-Reflection)]

- **ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model** (2026.03) \
  **描述**: 该论文提出由 VLM 引导的 JEPA 式潜在世界模型框架，通过双时间尺度设计结合密集帧动态预测与长程语义引导。它还引入分层金字塔表征提取模块，将多层 VLM 推理特征迁移到潜在预测中，以提升手部操作轨迹预测的鲁棒性。 \
  [[论文](https://arxiv.org/abs/2603.22281)]

- **Qwen3-VL Technical Report** (2025.11) \
  **描述**: 该报告介绍 Qwen3-VL，这是 Qwen 系列迄今能力最强的视觉语言模型，在广泛多模态基准上取得更优表现。它原生支持最高 256K token 的交错上下文，可无缝融合文本、图像和视频。 \
  [[论文](https://arxiv.org/abs/2511.21631)]
  [[项目](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list)]
  [[代码](https://github.com/QwenLM/Qwen3-VL)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen3-vl)]

- **InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency** (2025.08) \
  **描述**: 该论文提出 InternVL 3.5，一个新的开源多模态模型家族，在通用性、推理能力和推理效率上显著提升。其核心包括 Cascade Reinforcement Learning 框架，用于进一步增强多模态能力。 \
  [[论文](https://arxiv.org/abs/2508.18265)]

- **Qwen2.5-VL Technical Report** (2025.02) \
  **描述**: 该技术报告介绍 Qwen2.5-VL，一个旗舰视觉语言模型，具备更强的视觉识别、精确定位、文档解析和长视频理解能力。它还通过更好的 grounding 和结构化感知提升视觉环境中的智能体交互。 \
  [[论文](https://arxiv.org/abs/2502.13923)]
  [[代码](https://github.com/QwenLM/Qwen2.5-VL)]
  [[Hugging Face](https://huggingface.co/collections/Qwen/qwen25-vl-67ad1d2357b3028d1e9c4d56)]

- **InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling** (2024.12) \
  **描述**: 该论文介绍 InternVL 2.5，一个先进的多模态 LLM 系列，也是首个在 MMMU 基准上超过 70% 的开源 MLLM。它通过 Chain-of-Thought 推理带来 3.7 个百分点提升。 \
  [[论文](https://arxiv.org/abs/2412.05271)]
  [[Hugging Face](https://huggingface.co/spaces/OpenGVLab/InternVL)]

- **LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models** (2024.07) \
  **描述**: 该论文提出 LLaVA-NeXT-Interleave，同时处理多图像、多帧视频、多视角 3D 和多 patch 场景。它将视觉指令微调扩展到更复杂的多模态输入设置。 \
  [[论文](https://arxiv.org/abs/2407.07895)]
  [[项目](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)]
  [[代码](https://github.com/LLaVA-VL/LLaVA-NeXT)]

- **How Good is my Video LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs** (2024.05) \
  **描述**: 该论文提出 CVRR-ES，一个覆盖 11 类真实世界视频维度的视频 LMM 评测套件。它评估 9 个近期模型，并发现多数开源 Video-LMM 在复杂视频的鲁棒性和推理上仍然薄弱。 \
  [[论文](https://arxiv.org/abs/2405.03690)]
  [[项目](https://mbzuai-oryx.github.io/CVRR-Evaluation-Suite/)]

- **LLaVA: Visual Instruction Tuning** (2023.04) \
  **描述**: 该论文提出 LLaVA，一个使用机器生成指令微调数据端到端训练的大型多模态模型。它展现出强多模态对话能力，并在 Science QA 上取得当时最优结果。 \
  [[论文](https://arxiv.org/abs/2304.08485)]
  [[项目](https://llava-vl.github.io/)]

## 多模态推理
- **Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey** (2025.03) \
  **描述**: 这篇综述弥补了多模态大模型中多模态 Chain-of-Thought 推理缺少最新系统综述的问题，覆盖图像、视频、语音、音频、3D 和结构化数据。它给出基础定义、完整 taxonomy、跨应用方法分析，并总结未来多模态推理研究的开放挑战。 \
  [[论文](https://arxiv.org/abs/2503.12605)]
  [[项目](https://github.com/yaotingwangofficial/Awesome-MCoT)]

## 视觉-语言-动作
- **Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation** (2026.04) \
  **描述**: OneVL 面向 VLA 自动驾驶中的实时轨迹规划，将 Chain-of-Thought 推理压缩为由语言重建和未来帧预测共同监督的紧凑潜在 token。其三阶段训练流程让潜在推理在保持 answer-only 推理延迟的同时超过显式 CoT。 \
  [[论文](https://arxiv.org/abs/2604.18486)]
  [[项目](https://xiaomi-embodied-intelligence.github.io/OneVL/)]
  [[代码](https://github.com/xiaomi-research/onevl)]
  [[Hugging Face](https://huggingface.co/collections/xiaomi-research/onevl-models)]

# 嵌入模型

# 监督微调

## 监督微调方法
- **Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability** (2026.04) \
  **描述**: 该论文挑战“SFT 只会记忆而 RL 才能泛化”的常见说法，发现带长链式思维监督的推理 SFT 也能跨域泛化。其泛化效果取决于优化动态、训练数据和基础模型能力三者的共同作用。 \
  [[论文](https://arxiv.org/abs/2604.06628)]
  [[代码](https://github.com/Nebularaid2000/rethink_sft_generalization)]
  [[Hugging Face](https://huggingface.co/collections/jasonrqh/rethink-sft-generalization)]

- **ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection** (2026.01) \
  **描述**: 该论文提出 ProFit，一种监督微调方法，通过 token 概率作为语义重要性代理并屏蔽低概率 token，缓解单参考答案过拟合。该方法让训练更聚焦核心逻辑内容，并在推理和数学任务上优于标准 SFT 基线。 \
  [[论文](https://arxiv.org/abs/2601.09195)]

# 强化学习

## OPD
- **Self-Distilled RLVR** (2026.04) \
  **描述**: 该论文提出 Self-Distilled RLVR，将 on-policy distillation 与 RLVR 结合，由更大的教师模型为每条采样轨迹提供细粒度密集信号。该方法缓解了标准 RLVR 信号稀疏的问题。 \
  [[论文](https://arxiv.org/abs/2604.03128)]

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** (2026.02) \
  **描述**: 该论文证明 on-policy distillation 是 dense KL-constrained RL 的一个特例，并提出带灵活参考模型和奖励缩放因子的 G-OPD。其 reward extrapolation 版本 ExOPD 相比标准 OPD 更强，并能在融合 RL 训练的领域专家时让学生模型超越教师边界。 \
  [[论文](https://arxiv.org/abs/2602.12125)]
  [[代码](https://github.com/RUCBM/G-OPD)]
  [[Hugging Face](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data)]

## 奖励建模
- **MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels** (2026.03) \
  **描述**: 该论文提出 MemReward，一个基于图经验记忆的奖励预测框架，在有限标签下让 3B 和 1.5B 模型分别达到 Oracle 表现的 97.3% 和 96.6%。它还在域外任务上超过 Oracle。 \
  [[论文](https://arxiv.org/abs/2603.19310)]
  [[代码](https://github.com/ulab-uiuc/MemReward)]
  [[Hugging Face](https://huggingface.co/datasets/ulab-ai/MemReward)]

- **Scaling Reward Modeling without Human Supervision** (2026.03) \
  **描述**: 该论文研究无需人工标注的无监督奖励模型扩展，通过学习网页语料文档前缀与后缀之间的偏好来训练奖励模型。实验显示该方法在不同模型骨干上稳定提升 RewardBench，并改进 best-of-N 选择和策略优化。 \
  [[论文](https://arxiv.org/abs/2603.02225)]

- **Reward Modeling from Natural Language Human Feedback** (2026.01) \
  **描述**: 该论文将偏好数据上的 RLVR 用于训练生成式奖励模型，指出二分类任务会让 GRM 倾向于猜对结果而非给出可靠批判。论文提出方法缓解这一问题。 \
  [[论文](https://arxiv.org/abs/2601.07349)]

## 视频生成强化学习
- **KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration** (2026.05) \
  **描述**: KVPO 使用 ODE-native online GRPO 框架，将流式自回归视频生成器与人类偏好对齐。它用历史 KV 缓存的因果语义路由替代噪声探索，并基于 Trajectory Velocity Energy 优化速度场代理策略。 \
  [[论文](https://arxiv.org/abs/2605.14278)]
  [[项目](https://richard-zhang-ai.github.io/KVPO-Project/)]
  [[代码](https://github.com/Richard-Zhang-AI/KVPO)]
  [[Hugging Face](https://huggingface.co/Richard-ZZZZZ/KVPO)]

## 推理强化学习
- **All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models** (2026.04) \
  **描述**: 该论文提出 MUPO，一种强化学习方法，通过激励多解之间的发散思考来缓解 GRPO 训练 VLM 时的多样性坍塌。它让模型形成更深且更广的推理模式。 \
  [[论文](https://arxiv.org/abs/2604.00479)]
  [[项目](https://xytian1008.github.io/MUPO/)]
  [[代码](https://github.com/xytian1008/MUPO)]
  [[Hugging Face](https://huggingface.co/xytian1008/MUPO-Thinker-7B)]

- **The Art of Efficient Reasoning: Data, Reward, and Optimization** (2026.03) \
  **描述**: 该论文研究 LLM 的高效推理，通过 RL 激励短而准确的推理轨迹。论文总结了训练阶段、奖励设计和优化策略，并在 0.6B 到 30B 模型上分析泛化规律。 \
  [[论文](https://arxiv.org/abs/2602.20945)]
  [[项目](https://wutaiqiang.github.io/project/Art)]

- **FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization** (2026.03) \
  **描述**: 该论文提出 FIPO，一种强化学习算法，用于解决 LLM 推理瓶颈中的粗粒度信用分配问题。它针对 GRPO 式训练中结果奖励无法区分关键逻辑转折与普通 token 的问题进行改进。 \
  [[论文](https://arxiv.org/abs/2603.19835)]

- **Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis** (2026.02) \
  **描述**: 该论文提出 Agentic Proposing，一个使用专门智能体和 Multi-Granularity Policy Optimization 动态选择、组合模块化推理技能的框架。该框架用于合成高精度训练轨迹，从而增强大语言模型推理能力。 \
  [[论文](https://arxiv.org/abs/2602.03279)]

- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025.01) \
  **描述**: 该论文展示纯强化学习无需人工标注推理轨迹即可直接激发 LLM 的高级推理行为。所提出框架诱导自我反思、验证和自适应策略使用，并在数学、代码和 STEM 推理任务上取得强提升。 \
  [[论文](https://arxiv.org/abs/2501.12948)]
  [[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)]

## 智能体强化学习
- **Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning** (2026.05) \
  **描述**: Spreadsheet-RL 是一个用于在真实 Microsoft Excel 环境中训练专用表格智能体的强化学习微调框架，面向提示式智能体难以处理的复杂多步工作流。它结合自动化起止表格数据构建、多轮 Spreadsheet Gym 沙盒工具环境，以及 Domain-Spreadsheet 基准，以提升真实表格自动化能力。 \
  [[论文](https://arxiv.org/abs/2605.22642)]
  [[项目](https://spreadsheet-rl.github.io/)]
  [[代码](https://github.com/Spreadsheet-RL/Spreadsheet-RL)]
  [[Hugging Face](https://huggingface.co/datasets/Spreadsheet-RL/Spreadsheet-RL)]

- **Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration** (2026.02) \
  **描述**: 该论文提出 Actor-Refiner 协作机制，解决搜索增强推理 RL 中的多尺度信用分配问题。它缓解稀疏轨迹级奖励难以区分高质量推理与偶然猜对的问题，并减少冗余或误导性搜索行为。 \
  [[论文](https://arxiv.org/abs/2602.03647)]

- **Arena-RL: Training LLMs as Game Players with Vision-Language Action Models** (2026.01) \
  **描述**: 该论文提出 Arena-RL，一个通过视觉语言动作模型训练 LLM 驱动智能体玩视觉游戏的强化学习框架，重点从交互式游戏反馈中改进策略。实验表明，基于游戏轨迹的奖励优化能显著提升策略决策与跨游戏泛化。 \
  [[论文](https://arxiv.org/abs/2601.06487)]
  
- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025.03) \
  **描述**: 该论文提出 Search-R1，一个让 LLM 在逐步推理中通过 RL 学会自主生成搜索查询并使用实时检索的框架。它提升了模型获取外部知识和最新信息的能力。 \
  [[论文](https://arxiv.org/abs/2503.09516)]
  [[代码](https://github.com/PeterGriffinJin/Search-R1)]

- **Search-o1: Agentic Search-Enhanced Large Reasoning Models** (2025.01) \
  **描述**: 该论文提出 Search-o1，一个用智能体式检索增强生成机制和 Reason-in-Documents 模块增强大推理模型的框架。它通过精炼检索文档来缓解长程推理中的知识不足。 \
  [[论文](https://arxiv.org/abs/2501.05366)]
  [[代码](https://github.com/sunnynexus/Search-o1)]

## 视觉-语言-动作强化学习
- **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models** (2025.11) \
  **描述**: 该论文提出 SRPO，一个用于视觉语言动作模型的强化学习框架，用模型自身成功轨迹中的进展式奖励替代稀疏二元奖励。它利用潜在世界模型表征稳健衡量行为进展，并以更少 RL 步数在 LIBERO 上取得最优操作成功率。 \
  [[论文](https://arxiv.org/abs/2511.15605)]

# 智能体应用

## 工具调用
- **Thinking with Programming Vision: Towards a Unified View for Thinking with Images** (2025.12) \
  **描述**: 该论文指出当前多模态工具调用推理在简单图像旋转和损坏下仍然脆弱，并提出 CodeVision，一种让模型通过生成代码调用任意图像操作的 code-as-tool 框架。它结合 SFT、RL 和密集过程奖励，提升多工具推理、执行效率和错误恢复。 \
  [[论文](https://arxiv.org/abs/2512.03746)]
  [[代码](https://github.com/ByteDance-BandAI/CodeVision)]

## AI 研究
- **AI for Auto-Research: Roadmap & User Guide** (2026.05) \
  **描述**: 这篇综述从创造、写作、验证和传播等环节分析 AI 辅助科研，指出自动化在哪些地方可靠，以及自治系统在新颖性、实验和科学判断上仍会失败。它提供生命周期分类、基准套件、工具清单、设计原则和面向实践者的人类治理式 AI 研究工作流指南。 \
  [[论文](https://arxiv.org/abs/2605.18661)]
  [[项目](https://worldbench.github.io/awesome-ai-auto-research)]
  [[代码](https://github.com/worldbench/awesome-ai-auto-research)]

- **AIRA_2: Overcoming Bottlenecks in AI Research Agents** (2026.03) \
  **描述**: 该论文提出 AIRA_2，一种 AI 研究智能体架构，用于解决实验吞吐有限、基于噪声验证的选择不稳定以及单轮静态算子等瓶颈。它结合异步多 GPU worker、Hidden Consistent Evaluation 和交互式 ReAct 智能体，提升长程研究任务表现。 \
  [[论文](https://arxiv.org/abs/2603.26499)]

## 智能体技能
- **SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution** (2026.05) \
  **描述**: SkillsVote 将 Agent Skills 视为可复用经验资产，通过收集、推荐、归因和演化治理来缓解智能体轨迹噪声与难治理问题。它分析大规模开源技能语料，在执行前推荐结构化技能上下文，并只接纳证据门控的成功发现，以在不更新模型的情况下改进冻结智能体。 \
  [[论文](https://arxiv.org/abs/2605.18401)]
  [[项目](https://skills.vote/)]
  [[代码](https://github.com/MemTensor/skills-vote)]

- **From Context to Skills: Can Language Models Learn from Context Skillfully?** (2026.04) \
  **描述**: Ctx2Skill 面向长且密集上下文中的 context learning，解决人工标注技能成本高、自动构建技能缺少外部反馈的问题。它通过多智能体 self-play 与 Cross-time Replay 自动发现、细化并选择可复用自然语言技能，从而提升不同语言模型的上下文学习解题率。 \
  [[论文](https://arxiv.org/abs/2604.27660)]
  [[代码](https://github.com/S1s-Z/Ctx2Skill)]
  [[Hugging Face](https://huggingface.co/datasets/ssz1111/Ctx2Skill)]

- **SkillReducer: Optimizing LLM Agent Skills for Token Efficiency** (2026.03) \
  **描述**: 该论文提出 SkillReducer，一个两阶段优化框架，用于压缩 LLM 智能体技能这类预封装指令集。它在提升功能质量 2.8% 的同时，将技能描述和正文分别压缩 48% 和 39%，降低 token 成本与注意力稀释。 \
  [[论文](https://arxiv.org/abs/2603.29919)]

## 智能体开发
- **Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization** (2026.03) \
  **描述**: 该论文提出 Nurture-First Development，一种通过结构化对话而非固定代码优先或提示优先方式培养领域专家智能体的范式。它形式化了知识结晶循环、三层认知架构、双工作区模式和螺旋开发模型，用于持续把从业者隐性知识转化为可复用智能体资产。 \
  [[论文](https://arxiv.org/abs/2603.10808)]

## 记忆
- **Trajectory-Informed Memory Generation for Self-Improving Agent Systems** (2026.03) \
  **描述**: 该论文提出一个从 LLM 智能体执行轨迹中提取可行动经验，并在未来任务中作为上下文记忆检索的框架。它结合轨迹智能提取、决策归因、上下文学习生成和自适应记忆检索，尤其提升复杂 AppWorld 场景的任务完成率。 \
  [[论文](https://arxiv.org/abs/2603.10600)]

- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (2025.10) \
  **描述**: 该论文提出 BEAM，一个由长而连贯对话和探测问题组成的基准，用于评估 LLM 长期记忆；同时提出 LIGHT，一个包含情景记忆、工作记忆和草稿板的记忆框架。二者共同揭示长上下文记忆限制，并提升长程对话推理表现。 \
  [[论文](https://arxiv.org/abs/2510.27246)]

- **MLP Memory: A Retriever-Pretrained Memory for Large Language Models** (2025.08) \
  **描述**: 该论文提出 MLP Memory，一个轻量级参数模块，通过预训练 MLP 模仿 kNN 检索器行为来内化检索模式。该方法在 RAG 与微调之间架起桥梁。 \
  [[论文](https://arxiv.org/abs/2508.01832)]

- **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** (2025.07) \
  **描述**: 该论文提出 MemAgent，一个基于多轮对话 RL 的记忆智能体，用线性复杂度处理无限长文档。它旨在解决外推到超长上下文时性能退化的问题。 \
  [[论文](https://arxiv.org/abs/2507.02259)]
  [[项目](https://memagent-sialab.github.io/)]

# 视觉

## 目标检测
- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** (2022.03) \
  **描述**: DINO 通过对比去噪训练、用于锚点初始化的混合 query 选择，以及 look-forward-twice 框预测方案改进 DETR 类目标检测器。它在显著降低模型和数据需求的同时，在 COCO 上取得当时最优结果。 \
  [[论文](https://arxiv.org/abs/2203.03605)]
  [[代码](https://github.com/IDEA-Research/DINO)]

# 自动提示

## 提示优化
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025.07) \
  **描述**: GEPA 提出一种提示优化器，通过自然语言反思从试错中学习高层规则，平均超过 GRPO 6%，并最多减少 35 倍 rollout。它还比 MIPROv2 高出 10% 以上，并在代码优化的推理时搜索中展现潜力。 \
  [[论文](https://arxiv.org/abs/2507.19457)]
  [[代码](https://github.com/gepa-ai/gepa)]

## 评测器提示
- **Becoming Experienced Judges: Selective Test-Time Learning for Evaluators** (2025.12) \
  **描述**: 该论文提出 Learning While Evaluating，让 LLM-as-a-judge 系统在推理时通过自生成反馈更新元提示，从而按序列持续改进。它进一步提出 Selective LWE，只在自我不一致样本上更新，以更好的成本效率提升评测质量。 \
  [[论文](https://arxiv.org/abs/2512.06751)]

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **描述**: APE 通过从失败案例中自动发现辅助评测维度，并结合置信度感知选择进行集成，提升 LLM-as-a-judge 的可靠性。它更有效地使用测试时计算，从而提高与人类对齐基准的一致性。 \
  [[论文](https://arxiv.org/abs/2510.06538)]

# 笔记

## 论文解读

- **OPD：后训练中的能力整合接口** (2026-05-28) \
  **描述**: 从 Qwen3、GLM-5、MiMo-V2 与 DeepSeek-V4 的技术路径看 OPD 如何成为后训练中的能力整合接口。 \
  [[笔记](notes/zh/opd/post-training-opd.qmd)]
  [[English](notes/en/opd/post-training-opd.qmd)]

## 技术思考

暂无笔记。
