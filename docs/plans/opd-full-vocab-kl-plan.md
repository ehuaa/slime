# 档2 开发计划:V4 式全词表 OPD(On-Policy Distillation)

> 分支:`opd-full-vocab-kl` 工作目录:`/root/slime-opd`
> 状态:计划已确认,开发前先完成配套学习计划(见 `opd-learning-plan.md`)
> 预估:3.5~4.5 个工作日(不含 M5)
> 参考:DeepSeek V4 tech report(arXiv 2606.19348)式(29);slime 现有 OPD 实现 `loss.py:533`

---

## 0. 背景:slime 现状 vs V4 目标

### slime 现状(被 V4 点名批评的做法)

`slime/backends/megatron_utils/loss.py:533 apply_opd_kl_to_advantages`:

```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]   # 只在采样出的那个 token 上
advantages[i] = adv - args.opd_kl_coef * reverse_kl
```

teacher 只回传**采样 token 的 logp**(1 个标量/位置),用 `logπ_θ(y_t) − logπ_T(y_t)` 作为该位置
reverse-KL 的**单点蒙特卡洛估计**,以 advantage 惩罚的形式进入 PG loss。问题:估计方差高
(一个 token 代表整个分布),信号稀疏,V4 报告明确指出这会拖慢收敛。

### V4 目标(式 29)

```
L_OPD = Σᵢ wᵢ · D_KL(π_θ ‖ π_Tᵢ)
```

在学生**自采轨迹**的每个 response 位置上,对(近似)**全词表精确计算** KL 并直接作为 loss——
没有 reward、没有 advantage、没有 clip。位置级 KL 是解析值(方差为零),信息量是单点估计的
V 倍(V=词表大小)。

### 数学:top-K 分解(本方案的核心近似)

全词表 KL 按 teacher top-K 拆成三项:

```
D_KL(πθ‖πT) = Σ_v πθ(v)·(logπθ(v) − logπT(v))
            = −H(πθ)                                        ← 学生熵,精确、可微
              − Σ_{v∈topK} πθ(v)·logπT(v)                   ← cross 项,teacher top-K 上精确
              − (1 − Σ_{topK} πθ(v))·logπT_tail             ← 尾部近似项
```

尾部两种模式(`--opd-teacher-tail-mode`):

- `uniform`(默认):`logπT_tail = log(tail_mass/(V−K))`,teacher 尾部质量均匀摊到剩余词;
- `renorm`:丢弃尾部,teacher 在 top-K 内重归一(KL 有系统性低估但更稳定)。

K=128 时训练好的 LM top 质量通常 >0.999,近似误差 <1e-3 nats;K=V 时退化为**精确全词表 KL**
(用于单测对拍)。

### 梯度(M1 自定义 backward 用)

对学生 logits `z`(softmax 前)的标准解析结果:

```
∂D_KL/∂z_j = πθ(j) · (logπθ(j) − logπT(j) − D_KL)
```

自定义 autograd Function 用这个解析式,避免 autograd 展开 softmax 图导致的双倍显存。

---

## 1. 总体架构决策

### 决策 A:teacher 全 logits 不可缓存 → top-K 压缩

teacher 前向是**独立 pass**(`actor.py:463-474`:`_switch_model("teacher")` 权重切换,不能在
训练 microbatch 间穿插)。若缓存全 logits:GBS 256 × 平均 ~16K token ≈ 4M 位置 × 词表
102400 × bf16 ≈ **800GB/步**,不可行。压缩为每位置 `(topk_logps[K] bf16, topk_ids[K] int32,
tail_logmass fp32)`:4M × 128 × 6B ≈ **3GB**,走现有 CPU offload 路径(`actor.py:259`)。

### 决策 B:loss 侧只需学生分布,天然在训练前向里

训练 microbatch 的 forward 本来就产出学生 logits(vocab-parallel 分片),KL 三项全部可在
loss function 里由学生 logits + 缓存的 teacher top-K 算出,**训练时不需要 teacher 在场**。

### 决策 C:沿序列分块 + 解析梯度控显存

`[chunk, V]` 的 fp32 中间量按 chunk=1024 只有 ~0.4GB;现有 `--recompute-loss-function`
(loss.py:~1199 `checkpoint(func,...)`)可直接兜底。

### 决策 D:与现有 OPD 互斥,复用其 teacher 基建

`--opd-full-vocab-kl` 开启时跳过 `apply_opd_kl_to_advantages`(advantage 通道不动),loss 里加
KL 项。teacher 加载(`load_other_checkpoint("teacher", ...)`,actor.py:120-122)、权重切换
(`weights_backuper`)、`--opd-teacher-load/--opd-teacher-ckpt-step` 参数全部复用。

---

## 2. 里程碑详解

### M1 — vocab-parallel top-K KL 原语(~1 天)

**目标**:一个可微、TP 安全、分块的 `compute_topk_kl_from_logits()`。

**原理**:Megatron 的 lm_head 输出是 **vocab-parallel** 的——TP4 下每 rank 只持有词表 1/4
(25600 词)的 logits `[T, V/tp]`。任何涉及 softmax 归一化的量都需要跨 TP 组通信:

1. `logits_max`:`all_reduce(MAX)`(数值稳定);
2. `logZ = logsumexp`:局部 `sum(exp(z−max))` 后 `all_reduce(SUM)` 再取 log;
3. 熵项:现成的 `_VocabParallelEntropy`(`slime/utils/ppo_utils.py:162`)已实现该模式
   (三次 all_reduce:max / normalized_sum_exp / sum_softmax_times_logits),**直接复用**;
4. cross 项:teacher `topk_ids` 是**全局词 id**,每个 TP rank 用
   `(ids >= vocab_start) & (ids < vocab_end)` 掩出本分片的 id,`gather` 本地 logits,算
   `πθ(v)·logπT(v)` 局部和,`all_reduce(SUM)` 合并——与 `compute_log_probs`
   (ppo_utils.py:151)对采样 token 的处理是同一模式,从 1 个 id/位置推广到 K 个;
5. 尾部项:`Σ_topK πθ(v)` 在 4 中顺带得到,uniform 模式乘常数 `log(tail_mass/(V−K))`。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/utils/ppo_utils.py` | 新增 `_VocabParallelTopKKL(torch.autograd.Function)`(仿 `_VocabParallelEntropy`,fwd 手写 all_reduce,bwd 用解析梯度)+ 包装函数 `compute_topk_kl_from_logits(logits, topk_ids, topk_logps, tail_logmass, tp_group, tail_mode, chunk_size)`(仿 `calculate_log_probs_and_entropy`:649 的分块循环) |
| `tests/test_vocab_parallel_topk_kl.py`(新增) | ① 单进程无 TP:K=V 时与 `torch.distributions.kl_divergence` 逐位相等(atol 1e-5);② gloo 2/4 进程模拟 TP 分片与单进程一致;③ K=128 随机 logits 误差 <1e-3;④ fp64 `gradcheck`;⑤ 尾部模式边界(tail_mass→0 的 log 保护) |

**验收**:5 组单测全绿。

### M2 — teacher pass 产出 top-K(~1.5 天,涉及面最广)

**目标**:teacher 前向时每位置输出 `(topk_logps, topk_ids, tail_logmass)` 并沿现有数据管线
流到训练 microbatch。

**原理**:teacher pass 走 `actor.compute_log_prob(..., store_prefix="teacher_")` → model.py
forward-only 路径 → `get_log_probs_and_entropy`(loss.py:386)。加一条 top-K 支路:

1. **vocab-parallel top-K**:每 rank 对本地分片 `torch.topk(K)` → 局部候选(值 + 全局 id)
   → TP 组 `all_gather` 得 `[T, tp*K]` 候选 → 再 `topk(K)` 合并。全局 top-K ⊆ 各分片
   top-K 的并集,严格无损;
2. **归一化**:`logZ` 用与 logp 相同的 logsumexp all_reduce;`topk_logps = topk_logits − logZ`;
   `tail_logmass = log1p(−exp(logsumexp(topk_logps)))`(数值稳定写法);
3. 全程 `no_grad`,bf16 存值、int32 存 id;
4. **per-sample 切片 + CP 对齐**:`_extract_per_sample`(loss.py:296)按样本切并处理 CP
   (cp4 下一条序列切成 2×half-chunk 分布在各 rank,见
   `cp_utils.get_logits_and_tokens_offset_with_cp`);1-D logp 的切片逻辑对 2-D `[T, K]`
   同样适用(按第 0 维行切),但函数签名需泛化——**最易错的一块**;K=1 时新路径必须与旧
   `teacher_log_probs` 逐位一致(内建回归);
5. offload 与回传:`rollout_data["teacher_topk_logps"/"teacher_topk_ids"/"teacher_tail_logmass"]`。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/backends/megatron_utils/loss.py` | `get_log_probs_and_entropy`(:386)加 top-K 支路或新增旁路 `get_topk_logprobs`;`_extract_per_sample`(:296)支持 2-D 行切 |
| `slime/backends/megatron_utils/model.py` | forward-only 收集路径与 `forward_step` 的 `get_batch` 键表(:487-505)各加 3 个新键 |
| `slime/backends/megatron_utils/actor.py` | teacher 分支(:463-474)传 `with_teacher_topk=args.opd_teacher_topk`;CPU offload 键表(:259)加 3 个新键 |
| `slime/backends/megatron_utils/data.py` | `get_batch` 支持新键(2-D 的 padding/重排);日志聚合(:317-345)把新键加入跳过统计名单 |
| `slime/backends/megatron_utils/cp_utils.py` | 预计不改(offset 与维度无关),单测覆盖 |

**验收**:mini 配置(tp2/cp2 小模型)teacher pass 跑通;K=1 退化逐位一致(atol 1e-6);
K=128 时 `exp(logsumexp(topk_logps))+exp(tail_logmass)≈1`。

### M3 — loss 集成与参数(~0.5 天)

**原理**:`loss_function`(loss.py:1140)分发到 `policy_loss_function`;归一化统一走
`sum_of_sample_mean`(天然兼容 `--calculate-per-token-loss`)。KL 作为加法项,
`λ = --opd-kl-coef`。**纯 OPD** = 零 reward RM + 关 nonzero-std filter(advantage 全 0 →
pg 项恒 0,loss 只剩 KL),配置即可表达,不需要新 loss_type。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/utils/arguments.py` | OPD 参数组(:1038-1076)加 `--opd-full-vocab-kl` / `--opd-teacher-topk`(默认 128)/ `--opd-teacher-tail-mode`;校验区(:1681-1711):full-vocab 需 `--use-opd --opd-type megatron` |
| `slime/backends/megatron_utils/loss.py` | ① `use_opd` 钩子(:682)加 `and not args.opd_full_vocab_kl`;② `policy_loss_function` 内取 3 个 teacher 键,调 M1 原语,`loss += λ·sum_of_sample_mean(kl_per_token)`;③ 新指标 `opd_full_kl`、`opd_teacher_topk_mass`(近似覆盖率)进 logging dict(:1023 旁的模式) |

**验收**:参数校验单测;mini 一步训练出两个新指标;flag 关闭时全路径 bit-exact(回归)。

### M4 — 运行脚本 + 端到端冒烟(~1 天)

**原理(自蒸馏判据)**:teacher = 学生自己的 ckpt 时 πθ≡πT → 每位置 KL 解析值为 0 →
`opd_full_kl ≈ 0`(<1e-3)且 `grad_norm ≈ 0`。这是对整条链(TP 归约、top-K 合并、CP 切片、
loss 接线)最强的端到端判据。**K 扫描判据**:K∈{64,128,512,1024} 时 `opd_full_kl` 单调收敛,
128→1024 变化 <5% 说明 128 够用。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `scripts/run-deepseek-v2-opd.sh`(新增) | 基于 multinode 脚本:`--use-opd --opd-type megatron --opd-full-vocab-kl --opd-teacher-load <ckpt> --opd-kl-coef 1.0`;零分 RM;**去掉** dynamic-sampling-filter(否则全组零方差全被过滤);保留内存防护段 |
| `slime/rollout/rm_hub/zero_rm.py`(新增 ~10 行) | 恒返 0.0 的 custom_rm |

**验收**:16 卡真实拓扑自蒸馏一步:`opd_full_kl < 1e-3`、`grad_norm < 1e-2`;K 扫描收敛;
显存峰值不超基线 +2GB。
**注意**:teacher ckpt 需 Megatron 格式(arguments.py:1696 校验);若只有 HF 格式,评估对
teacher 路径放开 bridge 加载,必要时列 M4.5。

### M5 —(可选,后置)多 teacher 与 sglang 路线

- **5a 多 teacher**:teacher 基建循环化(`--opd-teacher-load` 多路径 + `--opd-teacher-weights`,
  `weights_backuper` 注册 teacher0/1/...,loss 加权求和)。单 teacher 需求下不做。
- **5b sglang top-K teacher**:teacher 引擎用 `top_logprobs_num` 直接吐 top-K,免训练侧
  teacher 常驻;代价是占 rollout 容量 + 需改 `sglang_rollout.py` 核心代码。仅当 Megatron
  teacher pass 时间(预计与 ref pass 同量级,每步 +4~5 分钟)不可接受时再评估。

---

## 3. 新增参数一览

| 参数 | 默认 | 说明 |
|---|---|---|
| `--opd-full-vocab-kl` | off | 启用全词表 KL loss;开启时 OPD 不再注入 advantage |
| `--opd-teacher-topk` | 128 | teacher 分布压缩宽度;=vocab 时精确(仅调试) |
| `--opd-teacher-tail-mode` | uniform | 尾部质量:uniform 均摊 / renorm 重归一 |
| `--opd-kl-coef`(复用) | — | KL loss 权重 λ |

## 4. 测试矩阵

| 层级 | 内容 | 里程碑 |
|---|---|---|
| 单元 | KL 对拍(K=V 精确/K=128 近似)、gradcheck、TP 一致性、尾部边界 | M1 |
| 单元 | K=1 退化 == 旧 teacher_log_probs;top-K 质量守恒;CP 2-D 切片 | M2 |
| 回归 | flag 关闭时全路径 bit-exact | M3 |
| 端到端 | 自蒸馏 KL≈0 & grad_norm≈0;K 扫描收敛;显存峰值 | M4 |

## 5. 风险清单

1. **CP 切片对齐**(M2)——概率最高的翻车点;K=1 回归测试兜底;
2. **teacher ckpt 格式**——bridge 加载对 teacher 路径是否直通,M4 前验证;
3. **单测无 16 卡拓扑**——gloo 多进程小词表模拟,真实拓扑靠 M4 自蒸馏兜底;
4. **bf16 数值**——归约统一 fp32(照 `_VocabParallelEntropy`),tail_logmass 加 clamp;
5. **teacher pass 时间**——top-K 增量 ~10-20%;瓶颈时考虑与 ref pass 合并或 M5b。

## 6. 时间估算

M1 1 天 → M2 1.5 天 → M3 0.5 天 → M4 1 天,合计 **3.5~4.5 天**(不含 M5)。
