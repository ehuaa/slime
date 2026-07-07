# 两周学习计划:从零(Transformer 都不熟)到能跟进档2(全词表 OPD)开发

> 前提画像:会**操作** sglang/slime(改配置、看日志、跑实验都很熟练),但不了解其内部原理;
> Transformer 结构、推理并行、训练并行、RL 数学都从零开始。
> 节奏:每天 ~3-4 小时(理论 : 动手 ≈ 1:1)。动手练习尽量锚定你自己的集群、日志和
> `run-deepseek-v2-multinode.sh` 配置——你已经"见过"所有现象,这两周是补上现象背后的原理。
> 诚实的预期管理:两周从零到"独立主导"档2开发不现实;目标定为**能逐条读懂
> `opd-full-vocab-kl-plan.md` 的每个技术决策、能评审实现、能独立完成 M1 级别的编码**。
> 更深的并行细节(PP 调度、ZeRO-3 等)标注为"够用即可",可跳过。

---

## 总览

| 阶段 | 天数 | 主题 | 对档2的作用 |
|---|---|---|---|
| 第一段 | D1-D3 | Transformer 与 GPT 从零(重中之重:logits/softmax/logprob/交叉熵) | KL 蒸馏的全部对象就是 logits |
| 第二段 | D4-D5 | 推理原理(你每天在看的 sglang 日志背后)+ 你的模型 DSV2 结构 | 理解 teacher 打分、top-K logprobs |
| 第三段 | D6-D8 | 训练与并行(反向传播/显存/TP/CP,重点 vocab-parallel) | M1/M2 的直接基础 |
| 第四段 | D9-D11 | RL 数学(策略梯度→PPO→GRPO/DAPO→KL/蒸馏) | 读懂你在跑的每个 flag + 档2动机 |
| 第五段 | D12-D14 | slime 代码走读 + 档2计划映射 + 编码热身 | 收口 |

如果中途进度落后,**不可砍**的核心链:D1-D3 的 softmax/logprob → D6-D7 的 vocab-parallel →
D9 的策略梯度 → D11 的 KL 蒸馏 → D13 的计划映射。其余都可压缩。

---

## 第一段:Transformer 与 GPT 从零(D1-D3)

### D1:注意力与 Transformer 结构(直觉层)

- 看:3Blue1Brown《GPT 是什么》+《注意力机制》可视化(先建直觉,共 ~1h)
  https://www.youtube.com/watch?v=wjZofJX0v4M 和 https://www.youtube.com/watch?v=eMlx5fFNoYc
- 玩:LLM 交互式 3D 可视化(把一个小 GPT 每一层点开看)https://bbycroft.net/llm
- 读:The Illustrated Transformer(经典图解)https://jalammar.github.io/illustrated-transformer/
- 自测:一个 decoder-only LLM 的一层由哪几块组成(attention + FFN/MoE + 残差 + norm)?
  Q/K/V 各是什么?causal mask 为什么存在?

### D2:从代码理解 GPT(Karpathy,两天中的第一天)

- 跟写:Karpathy《Let's build GPT from scratch》(2h 视频,**必做**,跟着敲)
  https://www.youtube.com/watch?v=kCc8FmEb1nY
  代码:https://github.com/karpathy/nanoGPT
- 重点盯:embedding → blocks → **lm_head 输出 logits `[T, V]`** → cross-entropy loss。
- 自测:logits、softmax、log_softmax、cross-entropy、perplexity 的关系,手写公式。

### D3:采样、logprob 与词表(档2的语言)

- 继续 nanoGPT:读 generate 部分(temperature、top-k 采样)。
- 重点概念(直接服务 OPD):
  - `logπ(y_t|context)` = log_softmax(logits)[y_t] ——slime 里所有 `log_probs` 都是它;
  - temperature=1 采样 = 按 softmax 概率抽词——你 rollout 的 `--rollout-temperature 1`;
  - 分布的熵 H(π)、两个分布的 KL(π‖π')——手写定义式;
  - "top-K 质量":排序后前 K 个词的概率和——档2用 K=128 压缩 teacher 分布的依据。
- 动手:用 nanoGPT(或任一 HF 小模型)取一个位置的 logits:①算熵;②算 top-128 概率和
  (亲眼确认 >0.99);③采 1 个词,对比"单点 logp 差"和"解析 KL"(两个略扰动的分布)——
  **这 20 行代码就是档2动机的全部**。
- 自测:为什么说"采样一个 token 估计 KL"是高方差的?

## 第二段:推理原理 + 你的模型(D4-D5)

### D4:推理系统:你每天看的日志背后

- 读:KV cache 概念(Playbook 或任意图解)+ vLLM PagedAttention 论文 §1-3
  https://arxiv.org/abs/2309.06180
- 读:sglang RadixAttention 论文(前缀缓存,你 router 实验里 `prefix_cache_hit_rate` 的来源)
  https://arxiv.org/abs/2312.07104
- 对照自己日志逐项解释(你有现成数据!):`Prefill batch / Decode batch`、`#running-req`、
  `#token / token usage`(KV 池占用)、`cuda graph: True`(为什么 decode 用 CUDA graph)、
  retract(KV 不够时逐出重排队)、`gen throughput` 为什么随 batch 增大而升、单请求速度为什么
  随 batch 增大而降——上周 router 实验的"尾速由同伴数决定"现在应该能自己推出来。
- 自测:为什么 decode 是带宽/延迟受限而 prefill 是算力受限?

### D5:并行 serving + DeepSeek-V2 结构(你训的模型)

- 读:DeepSeek-V2 论文 §2(MLA:KV 压成 512 维 latent,为什么省 KV;MoE:80 专家 topk7)
  https://arxiv.org/abs/2405.04434
- 概念:serving 侧 TP(权重切开、每层 allreduce)、EP(专家分布到不同卡、alltoall)、
  dp-attention(MLA 的 KV 不能按头切,attention 部分做 DP)——对照你 SGLANG_ARGS 里的
  `--sglang-ep-size 4 --sglang-enable-dp-attention --sglang-dp-size 4` 逐个说明。
- 动手:解释你自己脚本里 SGLANG_ARGS 的每一行(写中文注释),说不清的记下来。
- 自测:为什么 MLA 模型开 dp-attention 收益大?(KV 已经很小,TP 切 KV 不划算)

## 第三段:训练与并行(D6-D8)

> 主线资源:HuggingFace Ultra-Scale Playbook(现代训练并行讲得最好的免费资料)
> https://huggingface.co/spaces/nanotron/ultrascale-playbook

### D6:训练基础:反向传播、优化器与显存账

- 读:Playbook 开头两章(单卡训练的显存四大块:**参数/梯度/优化器状态/激活**;
  bf16 混合精度为什么要 fp32 master weights)
- 概念:backward 是 forward 的镜像(每个算子都有对应的梯度算子);Adam 的 m/v 状态是参数的
  2 倍 fp32;激活值随序列长度线性增长 → recompute(计算换显存)。
- 动手:算你的 32B 模型每卡显存账(对照脚本注释里的 21GB expert 复制、
  `--optimizer-cpu-offload` ~23GB/GPU、`--recompute-granularity full`),数字对上为止。
- 自测:为什么训练比推理贵那么多显存?(梯度+优化器状态+激活,推理只有权重+KV)

### D7:张量并行 TP 与 vocab-parallel(M1 的直接基础,本段最重要的一天)

- 读:Playbook TP 章;Megatron-LM 论文 §3(列切/行切,f/g 算子:forward 一次 allreduce,
  backward 一次镜像 allreduce——训练与推理 TP 的差别)https://arxiv.org/abs/1909.08053
- **重点**:lm_head 也被 TP 切 → logits 是 `[T, V/tp]` 的**词表分片**。凡是要"跨全词表"的
  量(softmax 分母 logZ、熵、KL)都必须 TP 组通信:max 一次 allreduce + sumexp 一次 allreduce。
- 读代码:`slime/utils/ppo_utils.py:151 compute_log_probs` 和 `:162 _VocabParallelEntropy`
  ——结合上面的知识逐行读,这两个函数就是档2 M1 要仿写的模板。
- 动手:两进程 gloo 写 30 行玩具:词表 8 切 2×4,手写 allreduce 版 log_softmax,与单进程
  `F.log_softmax` 对拍。
- 自测:vocab-parallel 下取"全局词 id=v 的概率"分几步?(判断 v 在哪个分片 → 本地 gather
  → allreduce)——这就是 M1 cross 项。

### D8:CP / SP / PP / EP 一日通(CP 精读,其余够用即可)

- 读:Playbook 对应各章 + Megatron CP 文档
  https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
- **CP 精读**(M2 的风险点):一条长序列切给 cp 组各 rank;为均衡 causal attention 的
  三角负载,序列切成 2×cp 块,每 rank 拿"第 i 块 + 倒数第 i 块"**两个不连续片段**——所以
  slime 里 per-sample 切片需要 `cp_utils.get_logits_and_tokens_offset_with_cp` 这种偏移
  计算。手算一个 8 token / cp2 的例子。
- EP(训练版):专家的梯度和优化器状态也按 EP 切;edp>1 时专家状态跨节点复制——这就是你们
  ep8→ep16 修 OOM 的原理。
- 动手:逐行解释你脚本 PERF_ARGS(tp4/sequence-parallel/pp1/cp4/ep16/etp1/recompute/
  max-tokens-per-gpu 16384),画 16 卡两张并行网格:非专家 tp4×cp4,专家 etp1×ep16。
- 自测:cp4 下 65468 token 每 rank 多少 token?为什么正好卡着 max-tokens-per-gpu?

## 第四段:RL 数学(D9-D11)

> 心法:LLM RL 只有一条主线:**∇E[回报] = E[∇logπ · advantage]**。所有算法都是在
> "advantage 怎么算"(baseline/组内归一化)和"步子怎么限制"(clip/KL)上做文章。

### D9:策略梯度从零推导

- 读:OpenAI Spinning Up Part 1-3(Part 3 逐行推导 log-derivative trick,**必读**)
  https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- 收藏当字典:Lilian Weng 策略梯度总览
  https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- 关键点:①∇E=E[∇logπ·R] 的推导;②减 baseline 不变期望、只降方差;③advantage=R−baseline。
- 动手(全部用你已有的数据回答):GRPO 组内减均值 = 用同 prompt 8 样本均值当 baseline
  → 所以 `rollout/rewards` 恒≈0;全对/全错组 advantage 全 0 → 零梯度 → 这就是
  nonzero-std filter 的数学依据。
- 自测:把上面两条讲给自己听,不看笔记。

### D10:PPO → GRPO → DAPO(你正在跑的所有 flag)

- 读:PPO 论文 §1-3 https://arxiv.org/abs/1707.06347;RLHF Book 的 GRPO 章
  https://rlhfbook.com;DeepSeekMath §4(GRPO 出处)https://arxiv.org/abs/2402.03300
- 精读:DAPO https://arxiv.org/abs/2503.14476 ——你脚本里的四个组件全在这篇里:
  Clip-Higher(eps-clip-high 0.28)/ 动态采样(os+filter)/ Token-Level Loss
  (--calculate-per-token-loss)/ Soft Overlong Punishment(dapo_overlong.py)。
- 动手:用 DAPO 的语言写 5 句话复盘你们的 27-step 长度失控事故(截断→0 分→零方差→被过滤
  →长样本只受正反馈);对照 `rm_hub/dapo_overlong.py` 与论文公式核对。
- 自测:ratio=π_new/π_old 为什么出现?clip 在防什么?为什么惩罚和超长过滤不能同时用?

### D11:KL 估计与蒸馏(档2的直接理论)

- 必读短文:John Schulman《Approximating KL Divergence》http://joschu.net/blog/kl-approx.html
  (slime 的 `--kl-loss-type low_var_kl` 就是文中 k3;单样本估计的方差问题一目了然)
- 读:知识蒸馏原点 Hinton §1-2(软标签比 one-hot 信息多)https://arxiv.org/abs/1503.02531
- 读:Thinking Machines《On-Policy Distillation》博客(slime 现有 OPD 的参考实现来源)
  https://thinkingmachines.ai/blog/on-policy-distillation/
- 读:V4 report OPD 节(式 29)https://arxiv.org/pdf/2606.19348;
  选读:GKD https://arxiv.org/abs/2306.13649(forward/reverse KL 的
  mode-covering/mode-seeking 差异)
- 动手:重做 D3 的 KL 实验但升级:对比 ①解析全词表 KL ②采样 token 单点估计(1000 次的
  均值和方差)③top-128 近似——三个数放一起,档2的动机就是这张图。
- 自测:reverse KL(student 在前)为什么适合对齐蒸馏?

## 第五段:slime 代码 + 收口(D12-D14)

### D12:slime 自顶向下走读

- 读:slime README/docs https://github.com/THUDM/slime
- 走读(在 `/root/slime-opd`):`train.py`(主循环编排)→ `slime/ray/rollout.py`
  (RolloutManager:router 启动:991、over-sampling 循环、`_post_process_rewards`:629)
  → `slime/rollout/sglang_rollout.py`(结合 D4 知识)。
- 动手:画数据流图:prompt → sglang 生成 → Sample(reward) → filter → rollout_data →
  microbatch → loss,标出每步所在文件。

### D13:megatron_utils 深读 + 档2计划映射

- 走读:`actor.py train_actor`:440(ref pass → teacher pass:463 → actor logp → adv →
  train)、`_switch_model`/weights_backuper(一份显存装三套权重:CPU 备份按需换入);
  `data.py` offload 键表:317;`loss.py` `get_log_probs_and_entropy`:386 →
  `_extract_per_sample`:296(用 D8 的 CP 知识)→ `policy_loss_function` → `loss_function`:1140。
- 动手:追踪 `teacher_log_probs` 完整生命周期:产生(actor.py:469)→ offload(:259)→
  get_batch(model.py:503)→ 消费(loss.py:557)。**这条链就是档2 M2 要复制的模板。**
- 收口:重读 `opd-full-vocab-kl-plan.md`,逐条映射:M1 三次 allreduce←D7;M2 2-D 切片
  风险←D8;top-K 压缩←D3+D6 显存账;自蒸馏判据←D11 KL(p‖p)=0。给计划提 ≥2 个问题。

### D14:编码热身(M1 的原型)

- 练习 1:单卡 `topk_kl_reference(student_logits, teacher_logits, K)`:精确 KL vs top-K
  近似,画 K∈{8,32,128,512} 误差曲线。
- 练习 2:gloo 2 进程把学生 logits 按词表切开,allreduce 版与单卡对拍——**这就是 M1 原型**。
- 练习 3(可选):mini 配置跑一步现有 OPD(teacher=自己),确认 `opd_reverse_kl≈0`
  ——提前在现有代码上验证 M4 的自蒸馏判据。

---

## 资源速查表

| 主题 | 链接 |
|---|---|
| GPT 直觉(视频) | https://www.youtube.com/watch?v=wjZofJX0v4M |
| Transformer 图解 | https://jalammar.github.io/illustrated-transformer/ |
| 从零写 GPT(必做) | https://www.youtube.com/watch?v=kCc8FmEb1nY + https://github.com/karpathy/nanoGPT |
| LLM 3D 可视化 | https://bbycroft.net/llm |
| PagedAttention / RadixAttention | https://arxiv.org/abs/2309.06180 / https://arxiv.org/abs/2312.07104 |
| DeepSeek-V2(你的模型) | https://arxiv.org/abs/2405.04434 |
| 并行训练主线 | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| TP / SP / PP | https://arxiv.org/abs/1909.08053 / 2205.05198 / 2104.04473 |
| CP 文档 | https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html |
| ZeRO / MoE | https://arxiv.org/abs/1910.02054 / 2401.06066 |
| RL 入门(必读) | https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html |
| 策略梯度字典 | https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ |
| PPO / GRPO / DAPO | https://arxiv.org/abs/1707.06347 / 2402.03300 / 2503.14476 |
| KL 估计(必读短文) | http://joschu.net/blog/kl-approx.html |
| RLHF Book | https://rlhfbook.com |
| 蒸馏 / OPD | https://arxiv.org/abs/1503.02531 / https://thinkingmachines.ai/blog/on-policy-distillation/ |
| V4 OPD | https://arxiv.org/pdf/2606.19348 |
| slime | https://github.com/THUDM/slime |
