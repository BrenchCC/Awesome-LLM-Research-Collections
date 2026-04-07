# Awesome-LLM-Research-Collections

> Tool Recommend for read paper faster: [幻觉翻译](https://hjfy.top/)

# Contents
- [Attention](#attention)
- [LLMs](#llms)
- [Multimodal LLMs](#multimodal-llms)
- [Embeddings](#embeddings)
- [Reinforcement Learning](#reinforcement-learning)
- [Agents Application](#agents-application)
- [Auto-Prompt](#auto-prompt)

# Attention

## Attention Residuals

- **Attention Residuals** (2026.03) \
  **Description**: This work replaces fixed residual accumulation with attention over previous layer outputs, enabling input-dependent depth-wise aggregation and reducing PreNorm-induced representation dilution. It also introduces Block AttnRes for scalable training with lower memory and communication overhead. \
  [[Paper](https://arxiv.org/abs/2603.15031)]
  [[Project](https://github.com/MoonshotAI/Attention-Residuals)]

## LLMs

- **GLM-5: from Vibe Coding to Agentic Engineering** (2026.02) \
  **Description**: GLM-5 is a next-generation foundation model targeting long-horizon agentic engineering, with reduced training and inference cost and preserved long-context capability. It introduces asynchronous RL infrastructure and agent RL algorithms to improve post-training efficiency and real-world coding performance. \
  [[Paper](https://arxiv.org/abs/2602.15763)]
  [[Project](https://z.ai/blog/glm-5)]
  [[Code](https://github.com/zai-org/GLM-5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-5)]]

- **GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models** (2025.08) \
  **Description**: GLM-4.5 introduces an open-source MoE foundation model with hybrid reasoning modes (thinking/direct response) to better support agentic, reasoning, and coding tasks. It combines large-scale pretraining and RL-based post-training, and releases both full and compact variants with strong benchmark performance. \
  [[Paper](https://arxiv.org/abs/2508.06471)]
  [[Code](https://github.com/zai-org/GLM-4.5)]
  [[Hugging Face](https://huggingface.co/zai-org/GLM-4.5)]]

## Multimodal LLMs

- **V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators** (2026.04) \
  **Description**: V-Reflection converts MLLMs from passive visual consumers to active interrogators through a think-then-look reflection mechanism that grounds each reasoning step in visual evidence. A two-stage distillation design improves fine-grained perception while keeping inference fully autoregressive and efficient. \
  [[Paper](https://arxiv.org/abs/2604.03307)]
  [[Project](https://idea-research.github.io/V-Reflection/)]
  [[Code](https://github.com/IDEA-Research/V-Reflection)]
  [[Hugging Face](https://huggingface.co/garlandchou/V-Reflection)]]

## Embeddings


## Reinforcement Learning

## Agents Application

## Auto-Prompt

- **Auto-Prompt Ensemble for LLM Judge** (2025.10) \
  **Description**: APE improves LLM-as-a-judge reliability by automatically discovering auxiliary evaluation dimensions from failure cases and ensembling them with confidence-aware selection. It boosts agreement with human-aligned benchmarks by using test-time computation more effectively. \
  [[Paper](https://arxiv.org/abs/2510.06538)]
