 why rl matters

### [cot](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=cot&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJjb3QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.QsMBuuhiylrqNdPFUYcYzAr2Xca1AsNC_HhIUWIR3rY&zhida_source=entity) pattern

[sft](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=sft&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJzZnQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.qZscmr2G_HgpEn8u1Y69YOt9UQfBQ8MbfJyUzMIiA8U&zhida_source=entity) 阶段有一个十分重要的职责， 那就是给 posttrain 的 cot pattern 进行定型，这几乎决定了模型在 posttrain 阶段所能达到的效果上限。

关于 cot pattern 是什么，以及为什么会有这么大的影响，我们不妨举个例子。给定一个 prompt：997 × 1003，会存在以下三种 cot pattern：

- good pattern：997 × 1003 = (1000 − 3)(1000 + 3) = 1000² − 3² = 999991
- mediocre pattern：997 × 1003 = 1003 × (900 + 90 + 7) = 902700 + 90270 + 7021 = 999991
- bad pattern：997 × 1003 = 999991

显然，三种 cot pattern 都能帮助不擅长大数计算的模型在这道题上取得正确的结果，但泛化到类似分布的数据呢？

- good pattern 大概率可以稳定做对；
- mediocre pattern 要执行多次乘法和加法，步骤更多，出错概率也会更高。即使做对了，它的 token efficiency 也会低的让人难受；
- 至于 bad pattern 那就更糟糕了，以我对大模型的了解，我怀疑把这条数据喂给一个小模型后，不管遇到 997 × 多少，它都会回复 999991。

因此，好的 cot pattern 就是会具有输出更短、准确率更高、更易泛化的特点。诚然，sft 带来的 cot pattern 会在后续的 rl 阶段有所改变 ，但很多有些优质的 pattern 并不是在 rl 阶段能轻易 explore 出来的。这就像在训 agent 任务时，如果不在 sft 里去教会模型一些好的思路的话，main agent 不会想到它可以开启一个sub\_agent 去处理局部工作，sub\_agent 也很难想到它可以和其他 sub\_agent 去交换信息。

换句话说，进入 agent 时代，我们绞尽脑汁构造和筛选的 environment trajectories，本质上仍是在延续 [reasoning](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=reasoning&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJyZWFzb25pbmciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.0bCrtFxogVYA5-XVwyn5bvwPkZ3UqdQfBlmYs1uE0Ng&zhida_source=entity) 时代的同一项工作：寻找更优质的 cot pattern。

### rl 的作用

话说回来，仅仅有好的 cot pattern 并不足以训出来一个好的模型，就像天马行空、思路宽泛的人未必真的健谈。人需要学会把自己的想法组织成一种好的表达方式，模型亦需要。

具体来说的话，pretrain 阶段所学到的知识是不同的， tokenizer 的词表是有差异的，所以即使给定同一个 cot pattern，不同的模型也会有不同的最佳表达方式。就比如，两个模型都学会了 99 × 99 = 99 × (100 - 1 ) = 9801 的 cot pattern，但在实际应用时：

- 强模型解题，99 × 99 = 99 × (100 - 1 ) = 9900 - 99 = 9801 ；
- 弱模型解题，99 × 99 = 99 × (100 - 1 ) = 99 × 100 - 99 × 1 = 9900 - 99 = 9900 - (100 - 1) = 9900 - 100 + 1 = 9800 + 1 = 9801。

绝大多数情况下，我们拿到的 cot pattern 都来自于更强的模型，或者是人工编辑出来的，往往具备“表述极简、思路极优雅”的特点。可问题在于，我们所要训练的模型，不具备足够强的能力去直接消化这种 pattern，模型需要结合自身的能力对 sft 阶段提供的 cot pattern 进行“[本土化改造](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=%E6%9C%AC%E5%9C%9F%E5%8C%96%E6%94%B9%E9%80%A0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiLmnKzlnJ_ljJbmlLnpgKAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.FMxv_qHufcxD7YKDYqUi6cNjx9mAKUYXToTtwAaUnyM&zhida_source=entity)”。

这正是 rl matter 的一个关键所在： **在给定一个 cot pattern 的情况下，通过 explore 足够多的 response，找到与它本身的“知识量， [tokenizer](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=2&q=tokenizer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJ0b2tlbml6ZXIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.1UudFnpInZ_6foId3XP2VOGsf2H1LEBIwApdDGJJn7Q&zhida_source=entity) ”最为契合的表达方式，强化这种表达方式形成自己的专属 cot pattern**。

- reasoning 任务上，sft 的 cot pattern 只会告诉模型做完题之后反思一下会更好，但具体要反思多少次效果最好且效率最高， 这便需要模型在 rl 中自己去摸索。同理，强模型在 rl 的时候，会发现 cot pattern 中的某些步骤可以跳过，弱模型则在 rl 的时候会发现 cot pattern 省略的步骤还是写出来会更好；
- agent 任务上，sft 数据会告诉模型有哪些 sub agent 可以使用，每个 sub\_agent 该在什么时机下使用，但具体怎么用便需要 model 在 rl 阶段自己摸索。1T 的模型一次调用 2 个 sub\_agent 便可以得到一个正确的 reward 信号，100B 的模型则需要调用 5 个 sub\_agent 才能得到一个正确的 reward 信号。不同的模型会在 rl 中找到最适合自己的 agent setting。

这也解释了一个现象：sft 后起点低的模型有可能 rl 后终点更高。我们所看到的 sft 起点低，大概率是这种 cot pattern 和模型能力不适应所导致的，并不是 sft 数据本身质量不行。当我们用 sota 模型的 response 进行 sft，便常常因为其 cot pattern 质量过高而导致 sft 指标不尽如人意。要验证一个 cot pattern 的潜力，往往是需要 sft + rl 共同协作的，也就是现在 posttrain 的标配流程。

除此之外，rl 某种程度上还有一点“[基因突变](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=%E5%9F%BA%E5%9B%A0%E7%AA%81%E5%8F%98&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiLln7rlm6DnqoHlj5giLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.qqFnnTvyfaYlgiJyhYvwqXV6x0KlKUuT67wkzFw8gek&zhida_source=entity)” \+ “进化”的味道。 **当算力足够大，rollout\_n 开的足够大，模型总能有机会在 rl 时 explore 到新的更优质的 cot pattern**。此时，好的 rl 算法 （如 GRPO）则需要充分把握这条高质量的数据，给予他足够强的学习信号让模型记住这个 pattern。

**除了对 cot pattern 进行本土化改造外，rl 更重要的意义便是让模型变得可控**。

传统理解中，大家可能觉着只要精准控制了 sft 的[数据分布](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiLmlbDmja7liIbluIMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.QuiybftepbfBcGwE4Hb2Z1iObhpyKsvDZr5FZSqQX38&zhida_source=entity)，便可以控制 sft 后的模型输出内容。遗憾的是，时代变了，由于大量合成语料进入到 pretrain / midtrain 阶段，posttrain 阶段出现乱七八糟的 pattern 的概率越来越大，即使我们所有的 sft 数据都很规范。

道理很简单，pretrain 阶段只看数据质量和数据干净程度，根本不会考虑 posttrain 阶段为了用户体验更好而加的一堆乱七八糟的规则。那些质量很高但格式非常不优雅的数据，训了几十 T，这怎么可能是随便几条 sft 数据能压制的，何况 sft 根本不具备打压 pattern 的能力。

sft 后的模型会输出多个 <think> 和 </think>，影响推理服务切分 cot 和 response；sft 后的模型中英混杂屡教不改，影响用户体验；sft 后的模型输出长度十分不稳定，有时候不反思，有时候反思几十次 …… 诸如此类，这些让产品头大的 case，在 [reward hack](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=reward+hack&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJyZXdhcmQgaGFjayIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI4MDQ5ODMyNywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.d3RIdIOePI7qvMSyv84ENn-oi6TPAKeJ4u63AROIXdY&zhida_source=entity) 面前却又是格外的稚嫩，随随便便在 rl 时候给个惩罚 loss 就再也不会出现了。

至于现在标配的“思考档位 / 思考深度”，作为“模型可控化输出”的重要模块，看似复杂，通过在 rl 阶段给予不同档位不同的上下文窗口便可轻松实现。

### rl 与蒸馏

既然我们在讨论 rl 的重要性，那就不可避免的要再提一下蒸馏（utilize better model response to sft）了，关于二者谁更 matter 的争吵从 2023 年一直持续到了现在。

目前没有直接证据能表明说 rl 后的模型相较于 sft 后的模型获得了某些质的提升，实际上， **rl 带来的收益是可以通过蒸馏而进行窃取的**。rl 本质还是在寻找更好的 pattern，通过堆算力找到的好 pattern 的确价值千金，但这个寻找过程可能并不具备价值。我们完全可以通过让 rl 后的模型当 teacher，借助蒸馏把它探索到的行为分布迁移到另一个模型上，从而得到一个指标基本接近 rl 模型的 student 。这也就是业界常使用的合版方案之一：对多个 rl 子模型进行reject sampling sft （另一个方案则是 [OPD](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=OPD&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJPUEQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.zGP0wKzS0S15n5B0Z_5uukPsIyGKCHBFx3RXc3a7bDY&zhida_source=entity)）。

当我们感慨某家模型的指标真高、token efficiency 真强的时候，很可能这个模型的的背后有着一个更强的 teacher 帮助它走了捷径。

很浅显的道理，1T model 在 rl 的时候 explore 到的 pattern，会比 100B model rl 时候 explore 到的 pattern 更加丰富。这个时候，把 1T model 找到的优质 pattern 喂给 100B model，再通过 rl 去适应这个 pattern 的具体表达方式（通常会让小模型的输出变得更长），小模型的指标就基本上和大模型的指标接近了 —— 这种做法在同尺寸、同词表模型上更是屡试不爽。

甚至，蒸馏是可以通过改变数据分布来让 student 比 teacher 更强的。例如，teacher 的 cot 是随机出中英文的，英文 cot 的整体质量更高，我们便可在蒸馏时强行只保留英文 cot 的数据，从而得到更高的指标 —— 但要注意，蒸馏或 OPD，任何能让 student 效果超过 teacher 效果的操作，都应该被用到去提升 teacher 效果上（要么是 student 用的数据质量更高，要么是 student 用的 reward 信号更准），而不是去思考这个操作有多厉害。

概括下来， **蒸馏是一种偷看学习笔记的行为**：无论是蒸馏 sota 模型的 response，还是蒸馏自家模型 rl 后的 response，都能通过让模型学到一个好的 pattern，得到一个起点很高且或潜力很大的 sft model。在此基础上， **通过 rl 去 adapt cot pattern 和 fix bad pattern，进一步消化笔记中的知识**，一个优秀的 posttrain model 便闪亮登场了。这也是为什么那么多小作坊选择只做 sft 不做 rl 的原因，又省算力、效果又好的方案，谁能拒绝呢？

一言以蔽之：认为蒸馏无用，大抵是没亲自训过大模型的人；认为蒸馏是 posttrain 的全部，大抵是只想做追赶者的人。

### 关于[软蒸馏](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=%E8%BD%AF%E8%92%B8%E9%A6%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiLova_okrjppo8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.HVBTsA6117gVAnm--428VgbOlg0Y1oNOqYGxUJctjc0&zhida_source=entity)

蒸馏聊完了，不妨再聊聊“软蒸馏”。

distill cot pattern 是最好用的蒸馏方法，但其实也是最低级的蒸馏方法。真正高级且常用的蒸馏方法，是通过借助 sota 模型的能力来优化我们自己的模型，包括但不限于：利用 [sota](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=4&q=sota&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJzb3RhIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjgwNDk4MzI3LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjQsInpkX3Rva2VuIjpudWxsfQ.uTpDJA89AjO-VBjPN73e4WqVH2SBExCFePnfrv-SvH0&zhida_source=entity) 模型合成现有模型能力边界的数据，利用 sota 模型分析现有模型的 pattern 缺点，利用 sota 模型优化 rl 时候所使用的 verifier ……

**只要国外模型领先于国内模型，那么国内模型就有源源不断的进步空间和优化手段。**

软蒸馏是普遍存在的现象，任何一家国内模型厂商都不敢说自己的优化过程中没有GPT/Claude的帮助，只能说革命尚未成功，差距依然巨大。

---

除了在 [infra](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=infra&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiJpbmZyYSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI4MDQ5ODMyNywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.xlHm50-ZQQFUFds9n2PciyFmVmy2T4fNEn1uIhXRBNo&zhida_source=entity) 上的[硬实力](https://zhida.zhihu.com/search?content_id=280498327&content_type=Article&match_order=1&q=%E7%A1%AC%E5%AE%9E%E5%8A%9B&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODY3MjU4MTUsInEiOiLnoazlrp7lipsiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODA0OTgzMjcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.t_5gUghO5XK78frZEXU7YCRF8nkhwj0IfN2xXD1QOas&zhida_source=entity)要拼刺刀外，llm 在纯训练阶段已经是明牌竞争了，比的就是谁更能 scaling：scaling model parameters，scaling rl data，scaling agent environments。

因此，越是想成为领头羊的团队，越是要加大对 rl 的投入，早日摆脱对 sota 模型的（软）蒸馏依赖。反过来也一样，如果只是追求模型在某个能力上的应用价值，蒸馏便已是最好的选择。
