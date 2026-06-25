# slime 多机 RL 训练 Qwen3-30B-A3B —— 原理与实战笔记

> 本文是一次完整实战(2 机 × 8 卡 GRPO 训练 Qwen3-30B-A3B)过程中,对 slime 源码、并行机制、配置参数的梳理,以及踩坑与结果总结。

---

## 目录
1. [On-Policy Distillation (OPD)](#1-on-policy-distillation-opd)
2. [并行切分与梯度传播](#2-并行切分与梯度传播)
3. [slime 配置参数详解](#3-slime-配置参数详解)
4. [本次 RL 训练全过程总结](#4-本次-rl-训练全过程总结)
5. [关键踩坑与经验](#5-关键踩坑与经验)

---

## 1. On-Policy Distillation (OPD)

OPD 让 student 在自己的 rollout 上学习,匹配 teacher 的 token 级 log-prob。它是**叠加在 advantage 上的一个 KL 惩罚项**,与 advantage estimator(GRPO/PPO/GSPO…)正交:

$$\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot \big(\log\pi_{\text{student}}(y_t) - \log\pi_{\text{teacher}}(y_t)\big)$$

逐 token 计算(`loss.py: apply_opd_kl_to_advantages`):`reverse_kl = student_log_probs - teacher_log_probs`,是 student→teacher 反向 KL 的**单样本 Monte-Carlo 估计**(只在采样到的 token 上)。

### 两种 teacher 模式

| | SGLang 模式 | Megatron 模式 |
|---|---|---|
| teacher 位置 | 独立的外部 SGLang server | 作为额外的**冻结** Megatron 模型加载进训练进程 |
| 何时打分 | rollout 阶段(reward_func 发 HTTP) | 训练 step 的 forward 阶段 |
| 架构要求 | **可不同**(进程隔离) | **必须同架构**、放得进显存 |
| 触发参数 | `--opd-type sglang` + `--rm-url` | `--opd-type megatron` + `--opd-teacher-load` |

**SGLang 模式如何绕开架构不匹配**:teacher 降格为"对 student 的 token 逐位打分的远程服务"。`reward_func` 发 `input_ids=sample.tokens`、`max_new_tokens=0`、`return_logprob=True` —— teacher **不生成,只 teacher-forcing 打分**。师生只交换 token id 和 log-prob,所有架构耦合被进程边界抹掉。**唯一硬约束:同一套 tokenizer/词表**(token id 必须指向相同 token)。teacher **可以**很大(这是 sglang 模式的卖点,大模型不占训练资源),但不**必须**很大——要求是"比 student 强"。

**Megatron 模式的本质**:teacher 和 reference model 同机制——`_switch_model("teacher")` 切进来 `compute_log_prob`(**纯 forward,no_grad,无 backward,无优化器**),再切回 actor。文档里 "teacher log-probs computed during the training forward pass" 的 "training forward pass" 指**训练 step 内部的前向计算**,**不是训练 teacher**。整个 step 只有 actor 走 backward。

### KL 怎么对齐(两端都裁到 response 区间)

采集侧(`post_process_rewards`):
```python
teacher_log_probs = torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], ...)
teacher_log_probs = t_log_prob[-response_length:]
```
- **为什么先 `[1:]` 再 `[-response_length:]`**:序列第 0 个 token 没有前文,SGLang 返回的 logprob 是 `None`。`[1:]` 在 **list 层**先把这个 `None` 去掉,否则 `torch.tensor([None, ...])` 在**构造时**就报 `TypeError`(`[-response_length:]` 是对已构造 tensor 的切片,来不及切掉 None)。`[1:]` 解决"None 不能进 tensor",`[-response_length:]` 解决"只要 response 区间"——两步各管各的。

训练侧:student 对同样 response token 算出 `student_log_probs`,逐 token 相减 → 加权进 advantage。对齐前提:**同词表 + 同 token 同位置**(两端都裁到 response 区间)。

---

## 2. 并行切分与梯度传播

### 2.1 DP 是自动算出来的,不用配

`megatron/core/parallel_state.py:741-746`:
```python
model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
data_parallel_size = world_size // model_size           # DP 推导,函数签名里没有 dp 参数
```
- `world_size = num_nodes × gpus_per_node`(ray/torchrun 拉起的进程数)。
- 例:`tp2, pp1, cp1` 在 8 卡上 → `DP = 8/2 = 4`。**整除是硬约束**,否则报错。
- 所以 `tp` 只决定"一份模型切几片",剩下的卡由 DP 吃掉。**4B 模型 tp2 在 8 卡上 = TP2 × DP4,8 卡全程在算。**

### 2.2 MoE 的两套 DP(非专家 / 专家分开)

同一批物理 GPU 被**两套并行视图**切分:
```python
data_parallel_size        = world_size // (tp · pp · cp)        # 非专家(attention/dense)
expert_data_parallel_size = world_size // (etp · ep · pp)       # 专家(MoE)
```
- 默认 `order="tp-cp-ep-dp-pp"`(以 pp 结尾)+ pp=1 时,允许**专家 DP ≠ 非专家 DP**。
- 本次 16 卡 `tp4, ep8`:非专家 dp = 16/4 = **4**,专家 edp = 16/8 = **2**。
- 不变量:`dp·tp·cp = edp·etp·ep = world/pp`(都 = 16)。**EP 不需要额外的卡——它把 attention 那批卡按"专家维度"重切了一遍**,层间靠 all-to-all(`--moe-token-dispatcher-type alltoall`)换布局。

### 2.3 权重怎么切分(以 node0 的 GPU0-7 为例)

| GPU0 上的权重 | 类型 | 谁持有相同副本 | 梯度同步组 |
|---|---|---|---|
| attention 分片0 | TP 分片(每卡半个模型) | GPU4, GPU8, GPU12 | DP 组 `{0,4,8,12}`(跨机) |
| expert 0-15 | EP 分片(各卡持不同 expert) | GPU8 | eDP 组 `{0,8}`(跨机) |

- **TP 组内 `0,1,2,3` 是不同分片,不是副本**,不互相做权重梯度同步。
- 物理落位(rank order tp 在最内层):**TP 落在单机内**(NVLink),**DP/eDP 跨机**(网络)。`ep8 = 单机卡数` → **MoE all-to-all 也在机内完成**(这是选 ep8 而非 ep16 的原因)。

### 2.4 梯度传播:激活梯度 vs 权重梯度

反向时有**两种梯度**走不同路径:

| | 是什么 | 流向 |
|---|---|---|
| **激活梯度 `dL/dx`** | loss 对每个 token 激活的梯度 | 沿计算图 expert→attention,**跨布局靠 all-to-all** |
| **权重梯度 `dL/dW`** | loss 对权重的梯度 | 沿各自 DP 组规约(attention→dp,expert→edp),**互不相通** |

- **attention 的权重梯度不依赖 MoE 的权重梯度**;但**依赖激活梯度 `dL/dh`**——它必须先反向**穿过 MoE 层的计算**(experts backward + 反向 all-to-all)才能算出。所以是"穿过 MoE 传回来的激活梯度张量",**不是** MoE 权重梯度,也**不是**标量 loss。
- expert↔attention 之间唯一的桥是 **all-to-all**(运 per-token 激活梯度,前向 dispatch 的镜像)。权重梯度的 DP/eDP 规约是两条独立支线,**两个 DP 组之间没有梯度流动**。

### 2.5 distributed optimizer (ZeRO-1) 与 reduce-scatter

多机默认开 distributed optimizer,把 "DP 组内 all-reduce + 各自全量更新" 换成更省显存的 **reduce-scatter + all-gather**:
1. **reduce-scatter**:DP 组内对梯度规约 + 按 DP 切片,每卡只拿 1/dp 的平均梯度;
2. 每卡只更新自己那 1/dp 参数(optimizer states 也只存 1/dp → **显存省 dp 倍,这是多机能去掉 CPU Adam 的原因**);
3. **all-gather**:把更新后的参数片拼回完整分片。

- 汇总的是**梯度不是 loss**(各卡数据不同 → 本地梯度不同 → 求平均);loss 不用预先汇总,因为梯度线性,平均推迟到梯度上做等价。
- **通信量**与 all-reduce 相同(`all-reduce = reduce-scatter + all-gather`),省的是**显存**,外加 optimizer step 的少量冗余计算。
- **ring reduce-scatter 没有中心规约卡**:第 k 块的规约结果落在 rank k,规约加法均摊到所有卡(每卡 ~`(P-1)·N/P` 次)。

---

## 3. slime 配置参数详解

### 3.1 数据 / batch

| 参数 | 单位 | 作用 |
|---|---|---|
| `--rollout-batch-size 32` | prompt | 每个 rollout 采几道题 |
| `--n-samples-per-prompt 8` | 回复/题 | 每题采几条(GRPO 组内算 advantage,必须 ≥2) |
| `--global-batch-size 256` | **样本** | 一次梯度更新用多少条样本(单位是 sample 不是 prompt) |
| `--num-rollout 100` | rollout 步 | 一共循环多少轮 |
| `--balance-data` | — | 按 token 数在 DP rank 间均衡(Karmarkar-Karp),减少 straggler |

- 一轮产出 = `rbs × n = 256` 条;`num_steps_per_rollout = rbs×n / GBS = 256/256 = 1` → **每轮训 1 步 = 完全 on-policy**。
- 两个 "epoch" 要分清:**外层**(过 prompt 数据集几遍,数据会 wrap+reshuffle)vs **内层**(同一批生成样本复用几次)。本次内层=1(用一次即弃);外层 `100×32/17398 ≈ 5.6` 遍。RL 反复的是**题目**,不是**答案**(on-policy 数据一更新就过时,不能反复用,这与 SFT/pretraining 的"固定标签可多 epoch"本质不同)。

### 3.2 性能 / 并行

- `--use-dynamic-batch-size` + `--max-tokens-per-gpu 20480`:micro-batch 不按固定条数,而按 **token 预算**贪心打包(first-fit,每个 ≤ `max_tokens × cp`)。长短不一的数据拼到接近预算,**计算/显存恒定,无 padding 浪费**。
- **micro-batch 是梯度累积单位,不是 optimizer step**:一个训练步内所有 micro-batch forward+backward**累积梯度**,期间**权重不变**,最后才 `optimizer.step()` 一次。所以 micro-batch 切分(含"超长样本独占一个 micro-batch")**对 on/off-policy 完全中立**——off-policy 只由"每 rollout 做几次 optimizer step"(GBS)和"训推同步时机"决定。
- 层级:`rollout(256) → 按 GBS 切训练步 → 步内按 token 打包成 K 个 micro-batch → 分给 dp_size 个 rank(每 rank K/dp = 梯度累积步数)`。

### 3.3 资源 / colocate

- `--colocate`:训推**共用同一批 GPU**,自动开 `--offload`(rollout 时 offload 训练态,反之亦然),时间片轮流。适合中小规模、卡不富裕。
- **训推分离**(235B 那套):去掉 colocate + `--rollout-num-gpus`,各占独立卡。超大模型 colocate 的 offload 搬运太贵,故分离;`--update-weight-buffer-size`(默认 512MB)是权重同步分块大小,大 MoE 调大更快。
- `--rollout-num-gpus-per-engine`:**引擎数 = rollout_gpus / per-engine**,多引擎挂同一 sgl-router 做请求级数据并行。`ep-size ≤ per-engine`。**RL rollout 要扩的是引擎数,不是把单引擎做大**(单大引擎跨机推理慢)。

### 3.4 reward / GRPO

- `--rm-type deepscaler`:**规则二值奖励**——取 `</think>` 后的答案、抽 `\boxed{}`、和 label 做数学等价判定(`grade_answer_mathd` / `grade_answer_sympy`),对=1 错/无答案=0。
- **`rollout/raw_reward` = 这些 0/1 的均值 = 答对比例**(0.76 = 256 条里 76% 答对)。
- **`rollout/rewards` = GRPO 组内中心化后的 advantage**(≈0,真正喂梯度的训练信号)。
- GRPO args:`--advantage-estimator grpo`、`--eps-clip 0.2 / --eps-clip-high 0.28`(DAPO 解耦 clip)、`--entropy-coef 0`、`--kl-loss-coef 0`。
- **`train/kl_loss` = 对 ref(初始模型)的 KL**:`loss = pg_loss + kl_loss_coef × kl_loss`,**coef=0 → 不进梯度,纯诊断**。RL 中它**单调上涨是正常的**(策略在离开出发点);只要伴随 reward↑/eval↑ 就是健康。崩溃前兆看 reward↓ / 熵坍缩 / grad_norm 尖刺 / **ppo_kl**(新旧策略,非 ref)爆涨。

### 3.5 关键监控指标

| 指标 | 健康 | 报警 |
|---|---|---|
| `rollout/raw_reward` | 稳定 ↑ | 横盘/崩塌 |
| `eval/aime` | 跟着 ↑ | reward 涨但它不涨(reward hacking) |
| `rollout/entropy` | 缓慢 ↓ | 骤降到 0(坍缩) |
| `train/grad_norm` | 平稳 | 尖刺/NaN |
| `train/ppo_kl` | 小而稳(本次=0,on-policy) | 突然爆涨 |
| `train/train_rollout_logprob_abs_diff` | 小(~0.017,训推对齐) | 变大 |
| `raw_response_length/...clip_ratio` | 低 | 高(大量截断) |

---

## 4. 本次 RL 训练全过程总结

### 配置
- **模型**:Qwen3-30B-A3B(128 expert,top-8);**任务**:DAPO-math-17k 训练,AIME-2024 eval。
- **资源**:2 机 × 8 = 16 卡,colocate,GRPO。
- **集群**:master(node IP `10.107.229.32`)+ worker(`/etc/mpi/hostfile`,`ssh -p 8081`);跨机 NCCL 自动走 **InfiniBand + GDRDMA**(无需手设 `NCCL_IB_HCA`)。
- **并行**:megatron `tp4 / ep8`(非专家 dp4、专家 dp2);sglang `per-engine 4 + ep4 + dp-attention dp4`(4 引擎,每机 2 个)。
- **长度**:`rollout-max-response-len 16384`、`eval-max-response-len 32768`(消除 AIME 截断);`max-tokens-per-gpu 20480`;`GBS 256`(1 step/rollout,on-policy)。
- 多机去掉 CPU Adam offload(distributed optimizer 已切分 optimizer state)。

### 流程
1. **转换**:HF → Megatron `torch_dist`(`tools/convert_hf_to_torch_dist.py`,底层用 mbridge),8 卡转一次,产物 57G。torch_dist 是**重切分友好格式**,8 卡转、16 卡训完全解耦。
2. **多机启动**:脚本在 master 起 ray head,SSH 拉起 worker join,等 16 卡就绪再 `ray job submit`。
3. **训练**:从初始模型开始 GRPO。

### 结果(100 步完整轨迹)

| step | eval/aime | 备注 |
|---|---|---|
| 0(基线) | 0.8125 | |
| 20 | 0.83125 | |
| 40 | 0.8354 | |
| **60** | **0.8417** | **峰值**(对应 `iter_0000059`) |
| 80 | 0.83125 | |
| 100 | 0.8292 | 最终(对应 `iter_0000099`) |

- **净提升:0.8125 → 0.829,约 +1.7 点**,温和上涨;峰值在 **step 60(0.842)**,之后回落到 ~0.83 平台。
- **raw_reward**:全程 ~0.66–0.91 波动、均值 ~0.79,**没突破 ~0.8 平台**;**train_rollout_logprob_abs_diff** 稳定 ~0.017;**grad_norm ~0.04、entropy ~0.24** 平稳无坍缩;**ppo_kl=0**(on-policy)。

**解读**:这是"小幅改善后进入平台",**不是崩**(eval 稳在 0.83-0.84、reward 稳在 ~0.8)。非单调 + ±0.01~0.02 抖动来自 **AIME 小样本(30 题)噪声 + 高基线平台效应**——基线就 0.81、很多简单题"全对"导致组内零方差、无 GRPO 信号。对这个强 base + 这份数据,该结果符合预期。

> **最优 checkpoint 是 `iter_0000059`(step 60,AIME 0.842),不是最终的 `iter_0000099`(step 100,0.829)**——step 60 正好是当初保留的续训点,两个都在盘上。要挑最好的模型用 step 60。

**想进一步提升的方向**(平台效应说明该改训练信号而非堆步数):dynamic sampling 过滤零方差组(最对症)> 数据难度过滤 / 换更难数据 > 加大 eval n-samples 降噪。

### 时间画像(每步)
- 稳态步(无 eval)≈ **8.5 min**:**生成占 ~92%**(rollout@16384 ~5.8min),train 计算仅 ~8%(~113s)。
- 逢 eval 步 ~24min(eval@32768 ~16min,30 题×16 samples 超长序列)。
- **瓶颈在生成侧**;100 步 ≈ 15 小时。

### 关于 mbridge / 免离线转换
- 训练时 `--megatron-to-hf-mode bridge` + `--load` 指向 HF 目录,可**跳过离线转换**(`checkpoint.py:_load_checkpoint_hf` 用 `megatron.bridge` 直载),前提是该架构在 `megatron.bridge` 已注册(Qwen3MoeForCausalLM 在列)。但大模型/多机/反复起作业仍推荐先转 torch_dist(加载更快更稳)。

---

## 5. 关键踩坑与经验

### 坑 1:磁盘配额(转换时)
- `/mnt/zj-gpfs/model` 100% 满 → 转换写 torch_dist 时 `[Errno 122] Disk quota exceeded`。
- **解决**:换到有空间的盘 `/mnt/data/data`(改 `SAVE_DIR`)。

### 坑 2:torch.compile 融合交叉熵 CUDA 崩溃 ⭐
- 现象:首个 train step `torch.AcceleratorError: CUDA error: invalid argument`。
- 定位:`fused_vocab_parallel_cross_entropy → calculate_cross_entropy_loss`(`@jit_fuser` = torch.compile,见 `megatron/core/jit.py`)。
- **根因**:`--use-dynamic-batch-size` 使每个 micro-batch 的 `num_tokens` 不同 → torch.compile 转**动态 shape**;在**顶满 20480** 那档(logits `20480 × (152064/4=38016) = 778,567,680` 元素)inductor/triton kernel 启动配置非法。不是"模板不匹配",是动态 shape 最大档撞到 CUDA 启动限制。
- **修复**:`RUNTIME_ENV_JSON` 加 `TORCHDYNAMO_DISABLE=1`(+`TORCH_COMPILE_DISABLE=1`)→ 交叉熵走 eager(eager 对大张量没那个限制),`max_tokens_per_gpu` 可保持 20480。替代:降 `max-tokens-per-gpu`。

### 坑 3:checkpoint 太大撑爆配额(训练中)⭐
- **每个训练 checkpoint ~399 GB**(30B + distributed optimizer 状态),`--save-interval 20` 且 slime 不自动删旧的 → 堆到 1.5T,step 80 存盘时超配额崩溃(训练本身健康)。
- **修复**:删旧 checkpoint(留续训点)、`--save-interval` 20→50(续训期间只在末尾存一次);**从 `iter_0000059`(step 60)续训**——`--load` 指向 slime 目录会读 `latest_checkpointed_iteration.txt` 自动续。

### 经验:eval/rollout 长度对结果影响巨大
- `eval-max-response-len` 16384 时 **31% 截断** → eval/aime 被压到 0.66;放到 32768 后截断 1.9% → 真实水平 0.81。AIME 推理 median CoT ~12.5k token,**eval 长度不够会严重低估能力**。
- `rollout-max-response-len` 8192→16384 同理:太短会让训练回复被截断、reward 信号被长度而非对错主导。

### 经验:高基线下的零方差组
- 基线 reward ~0.76 偏高 → 不少 prompt 的 8 条全对/全错 → 组内 advantage=0,无学习信号。
- 若 reward 长期停滞,可开 **dynamic sampling**(`--over-sampling-batch-size` + `--dynamic-sampling-filter-path ...check_reward_nonzero_std` + `--partial-rollout`)只保留有信号的组,优先级高于单纯加 `n-samples`(后者线性涨成本且救不了太简单/太难的题)。

### 经验:多机 NCCL 网卡
- 复杂多 bond + RoCE/IB 节点,**优先让 NCCL 自动探测**(本次自动选对了 IB/GDRDMA)。若跨机 NCCL 在 init 处 hang,再设 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`。

### 经验:tp/ep 尽量对齐单机卡数
- 让通信最重的 TP 和 MoE all-to-all 压在机内(NVLink),只让 DP 跨机。`ep = 单机卡数(8)` 时 MoE all-to-all 不跨机。
