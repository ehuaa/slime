# 档2 开发计划:V4 式全词表 OPD(On-Policy Distillation)

> 分支:`opd-full-vocab-kl`
> 状态:v2(2026-07-13 修订)。配套学习计划见 `opd-learning-plan.md`(两周,先学后做或
> 边学边做;学习计划的 M1/M2 映射已随本次 v2 同步修订)。
> 预估:开发 2.5~3 个工作日(v1 为 3.5~4.5;M1/M2 依据 ms-swift 先例与 R3 实践简化),不含学习期
> 参考:DeepSeek V4 tech report(arXiv 2606.19348)式(29);slime 现有 OPD `loss.py:533`;
> **ms-swift PR #9678**(modelscope/ms-swift,V4 式 MOPD + top-K 蒸馏的先行实现,分析见 §0.5)

## 边界:已完成的档1 vs 本计划的档2(勿混淆)

**已完成并投产的只是档1**(2026-07-12,commit 4b164e13,`scripts/run-deepseek-v2-opd.sh`):
**token 级**纯 OPD——teacher 只对每个**采样出的 token** 给一个 logp,以
`−(logπθ(y_t) − logπT(y_t))` 注入 advantage(正是 §0 里被 V4 批评的单点蒙特卡洛估计,
只是配置成了零 reward 的"纯蒸馏"形态)。**没有任何 full-vocab 成分,也没有 MOPD**。

**本计划(档2)要开发的全部尚未动工**:M1 top-K KL 原语、M2 teacher top-K 管线、
M3 loss 集成、M4 验收,以及可选的 M5a 多 teacher(MOPD)。

档1投产对档2的价值仅限于**基建已验证**:teacher HF-bridge 加载、weights_backuper 权重
切换、zero RM、eval 覆盖、4 机 colocate——这些是档2直接复用的地基,不是档2本身。

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

### 数学:top-K 分解(v2 注:默认实现 M1a 取其 renorm 特例,完整分解属 M1b)

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

### 梯度(v2 后仅 M1b 精确分解的自定义 backward 需要;M1a 走自动微分)

对学生 logits `z`(softmax 前)的标准解析结果:

```
∂D_KL/∂z_j = πθ(j) · (logπθ(j) − logπT(j) − D_KL)
```

自定义 autograd Function 用这个解析式,避免 autograd 展开 softmax 图导致的双倍显存。

---

## 0.5 先行实现分析:ms-swift PR #9678(2026-07 修订新增)

ms-swift 在 PR #9678 实现了同一目标,三条与本计划直接相关的结论:

**a. 他们的"full-vocab"实际是 top-K 重归一 KL,没有尾部项、没有自定义 autograd。**
`gkd_loss.py`:student logits 在 teacher top-K 索引处 `gather` 出 `[N, K]`,**两边都只在 K 列
上 `log_softmax`(即在 top-K 内重归一)**,然后逐位置 KL/JSD、按 512 位置分块累加。K 列小图
直接走 PyTorch 自动微分。这正是本计划 §0 数学里的 `renorm` 尾部模式——被他们选为唯一实现并
投产。启示:**把 renorm 模式提为 M1 默认实现**(gather 之后不再需要任何 vocab-parallel 数学
与解析梯度),精确三项分解(uniform 尾部 + 解析 backward)降级为可选的 M1b 对照实验。

**b. 他们的 MOPD 是"按 tag 硬路由"的多 teacher,不是加权 KL 混合。**
`teacher_model_server` 接受 JSON `[{url, tags}]`,每个样本按 tag(数据集名或样本列
`--teacher_tag_key`)路由到**恰好一个** teacher;校验 tag 非空、跨 teacher 不重叠、匹配失败
即刻报错;多 teacher 并发推理。这与 V4 的领域专家 teacher 用法一致,M5a 采纳此设计
(替代原计划的 `--opd-teacher-weights` 加权求和;权重混合留作扩展)。

**c. 值得逐条搬运的工程细节**(计入 M1/M3 验收):
- teacher top-K 全为 -inf 的位置直接丢弃(覆盖率保护,`extract_active` 的 `uncovered` 分支);
- CP rank 分到空分区时返回零标量,保证后续 all-reduce 不挂死;
- 蒸馏温度缩放(两侧 logits 除以 T)与 β 旋钮(0=正向 KL,1=反向 KL,中间=JSD 混合);
- `tests/utils/test_multi_teacher.py`(405 行)是 M5a 路由测试的现成模板。

**与我们的差异**(我们的方案保留的优势):他们的 teacher 是外部 vLLM API(HF 路径为主,
Megatron 侧 TP-aware 钩子只在本地全 logits 时启用);我们是训练进程内 megatron teacher
(权重换入、无 API 往返、与 R3 协同),top-K 在训练侧一次前向内产出,不占推理容量。

---

## 1. 总体架构决策

### 决策 A:teacher 全 logits 不可缓存 → top-K 压缩 + 微批 record/pop 传递(v2 修订)

teacher 前向是**独立 pass**(`actor.py:463-474`:`_switch_model("teacher")` 权重切换,不能在
训练 microbatch 间穿插)。若缓存全 logits:GBS 512 × 平均 ~16K token ≈ 8M 位置 × 词表
128256 × bf16 ≈ **2TB/步**,不可行。压缩为每位置 `(topk_logps[K] bf16, topk_ids[K] int32)`:
8M × 128 × 6B ≈ **6GB**。

**传递方式(v2 关键修订)**:不走 v1 计划的 rollout_data 管线(per-sample 切片 + CP 对齐
+ data.py 键表,原风险清单第 1 条"最易翻车"),改用 **RoutingReplay 同款 record/pop 模式**
(`slime/utils/routing_replay.py`,已被 R3 在生产验证):teacher 的 `compute_log_prob` pass
与 student 训练前向消费**同一个 data_iterator、同一微批顺序与布局**(`fill_routing_replay`
actor.py:295 已证明该不变量,含 `num_steps_per_rollout>1` 的 `sum(num_microbatches)` 情形)。
teacher pass 逐微批把 `[T_mb, K]` 的 (logps, ids) 写入 pinned CPU 缓冲,student 训练前向的
loss function 逐微批 pop 回 GPU。微批张量本身已完成 CP 切片与 padding,**完全绕开
per-sample 2-D 切片和 cp_utils 对齐**。

### 决策 B:loss 侧只需学生分布,天然在训练前向里

训练 microbatch 的 forward 本来就产出学生 logits(vocab-parallel 分片),KL 三项全部可在
loss function 里由学生 logits + 缓存的 teacher top-K 算出,**训练时不需要 teacher 在场**。

### 决策 C:沿序列分块 + 解析梯度控显存

`[chunk, V]` 的 fp32 中间量按 chunk=1024 只有 ~0.4GB;现有 `--recompute-loss-function`
(loss.py:~1199 `checkpoint(func,...)`)可直接兜底。

### 决策 D:与现有 OPD 互斥,复用其 teacher 基建(已投产验证)

`--opd-full-vocab-kl` 开启时跳过 `apply_opd_kl_to_advantages`(advantage 通道不动),loss 里加
KL 项。teacher 加载(`load_other_checkpoint("teacher", ...)`,actor.py:120-122)、权重切换
(`weights_backuper`)、`--opd-teacher-load/--opd-teacher-ckpt-step` 参数全部复用——
**这条基建已由档1生产 run 验证**(2026-07-12 冒烟 + 投产:HF 目录经 bridge 直载 teacher 成功,
`latest_checkpointed_iteration.txt` 校验仅为 logger.info 不阻断;R3 与 teacher/ref 前向的
fallthrough 隔离由上游处理,actor.py:466)。v1 风险清单第 2 条已消除。

---

## 2. 里程碑详解

### M1 — top-K KL 原语(~0.5 天;v2 按 ms-swift 先例简化)

**目标(M1a,默认实现 = renorm 模式,ms-swift 同款)**:
`compute_topk_kl_from_logits(logits, topk_ids, topk_logps, tp_group, chunk_size, beta, temperature)`。

1. **唯一的 vocab-parallel 步骤是 gather**:teacher `topk_ids` 是全局词 id,每个 TP rank 用
   `(ids >= vocab_start) & (ids < vocab_end)` 掩出本分片 id,gather 本地 logits,
   `all_reduce(SUM)` 拼出完整的 `[T, K]` student logits(与 `compute_log_probs`
   ppo_utils.py:151 对采样 token 的处理同模式,1 个 id/位置推广到 K 个)。该步须可微
   (gather + masked all_reduce 的 autograd 天然成立,梯度散射回本地分片);
2. gather 之后是**纯本地小图**:两侧 `[T, K]` 除以温度 → K 列上 `log_softmax`(top-K 内
   重归一)→ 逐位置 KL(β 旋钮:0=正向,1=反向,中间=JSD);**不需要自定义 autograd
   Function、不需要熵/logZ 的全词表归约**;
3. 按 512~1024 位置分块累加(照 ms-swift `jsd_loss` 的 chunk 循环);分块为空(CP rank
   空分区)时返回零标量,保证后续 collective 不挂死;
4. teacher top-K 全为 -inf 的位置丢弃(覆盖率保护)。

**M1b(可选对照,+1 天,原 v1 方案)**:精确三项分解(复用 `_VocabParallelEntropy`
ppo_utils.py:162 的熵项 + cross 项 + uniform 尾部)+ 解析梯度
`∂D_KL/∂z_j = πθ(j)(logπθ(j) − logπT(j) − D_KL)` 的自定义 Function。仅当 renorm 的系统性
低估在 K 扫描(M4)中被证明影响训练时再做。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/utils/ppo_utils.py` | 新增 `compute_topk_kl_from_logits`(M1a:vocab-parallel gather + 本地 renorm KL,分块) |
| `tests/test_topk_kl.py`(新增) | ① 单进程无 TP:K=V 且 β=1 时与 `torch.distributions.kl_divergence` 逐位相等;② gloo 2/4 进程模拟 TP 分片与单进程一致;③ 空分区零标量;④ 全 -inf 位置丢弃;⑤ β/温度边界 |

**验收**:单测全绿。

### M2 — teacher pass 产出 top-K,record/pop 传递(~1 天;v2 改道,风险大降)

**目标**:teacher 前向逐微批产出 `(topk_logps, topk_ids)` 存 pinned CPU;student 训练前向
逐微批取回。**不再经过 rollout_data / per-sample 切片 / data.py 键表**(v1 的最高风险点整体
移除,原理见决策 A)。

1. **vocab-parallel top-K**(teacher 侧,`no_grad`):每 rank 对本地分片 `torch.topk(K)` →
   局部候选(值 + 全局 id)→ TP 组 `all_gather` 得 `[T, tp*K]` → 再 `topk(K)` 合并。全局
   top-K ⊆ 各分片 top-K 并集,严格无损;`logZ` 用 logsumexp all_reduce,
   `topk_logps = topk_logits − logZ`;bf16 存值、int32 存 id;
2. **TeacherTopkReplay**(仿 `RoutingReplay`,routing_replay.py):teacher 的
   `compute_log_prob` pass 中每微批 `record((topk_logps, topk_ids))` 到 pinned CPU;
   student 训练前向的 loss function 中 `pop_forward()` 回 GPU;微批顺序不变量与
   `fill_routing_replay`(actor.py:295)相同,`num_steps_per_rollout>1` 时按
   `sum(num_microbatches)` 全量录制;每 rollout 结束 `clear_all()`;
3. 内存账:GBS 512 × ~16K token ≈ 8M 位置 × 128 × 6B ≈ **6GB pinned host**/rollout
   (R3 路由数据同路径 ~45GB 已投产,量级更小);
4. **K=1 回归**:K=1 时 pop 出的 logps 必须与现有 `teacher_log_probs`(采样 token 路径)
   在采样 token 上逐位一致(内建断言,atol 1e-6)。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/utils/routing_replay.py` 或新文件 | `TeacherTopkReplay`(record/pop/clear,pinned CPU 缓冲,仿 `RoutingReplay`) |
| `slime/backends/megatron_utils/loss.py` | `get_log_probs_and_entropy`(:386)加 top-K 旁路(仅 teacher pass 且 `args.opd_full_vocab_kl` 时启用,算完即 record,不进返回 dict) |
| `slime/backends/megatron_utils/actor.py` | teacher 分支(:463-474)置 replay 录制开关;训练分支置消费开关;rollout 边界 clear |

**验收**:mini 配置(tp2/cp2 小模型)teacher pass 跑通;K=1 退化逐位一致(atol 1e-6);
K=128 时 `exp(logsumexp(topk_logps)) ≈ top-K 质量 ∈ (0.95, 1]` 且逐微批 shape 断言全过。

### M3 — loss 集成与参数(~0.5 天)

**原理**:`loss_function`(loss.py:1140)分发到 `policy_loss_function`;归一化统一走
`sum_of_sample_mean`(天然兼容 `--calculate-per-token-loss`)。KL 作为加法项,
`λ = --opd-kl-coef`。**纯 OPD** = 零 reward RM + 关 nonzero-std filter(advantage 全 0 →
pg 项恒 0,loss 只剩 KL),配置即可表达,不需要新 loss_type。

**文件改动**:

| 文件 | 改动 |
|---|---|
| `slime/utils/arguments.py` | OPD 参数组(:1038-1076)加 `--opd-full-vocab-kl` / `--opd-teacher-topk`(默认 128)/ `--opd-distill-beta`(默认 1=反向 KL,对齐 V4 式29;0=正向,中间=JSD)/ `--opd-distill-temperature`(默认 1);校验区(:1681-1711):full-vocab 需 `--use-opd --opd-type megatron`(`--opd-teacher-tail-mode` 移到 M1b 一并做) |
| `slime/backends/megatron_utils/loss.py` | ① `use_opd` 钩子(:682)加 `and not args.opd_full_vocab_kl`;② `policy_loss_function` 内取 3 个 teacher 键,调 M1 原语,`loss += λ·sum_of_sample_mean(kl_per_token)`;③ 新指标 `opd_full_kl`、`opd_teacher_topk_mass`(近似覆盖率)进 logging dict(:1023 旁的模式) |

**验收**:参数校验单测;mini 一步训练出两个新指标;flag 关闭时全路径 bit-exact(回归)。

### M4 — 运行脚本 + 端到端冒烟(~0.5 天;v2:脚本与零分 RM 已随档1存在)

**原理(自蒸馏判据)**:teacher = 学生自己的 ckpt 时 πθ≡πT → 每位置 KL 解析值为 0 →
`opd_full_kl ≈ 0`(<1e-3)且 `grad_norm ≈ 0`。这是对整条链(TP 归约、top-K 合并、CP 切片、
loss 接线)最强的端到端判据。**K 扫描判据**:K∈{64,128,512,1024} 时 `opd_full_kl` 单调收敛,
128→1024 变化 <5% 说明 128 够用。

**文件改动**(v2:脚本与零分 RM **已存在**,见档1投产,只需增量):

| 文件 | 改动 |
|---|---|
| `scripts/run-deepseek-v2-opd.sh`(已存在) | 加 `--opd-full-vocab-kl --opd-teacher-topk 128`(其余:megatron teacher、`--rm-type zero`、无过滤器、eval yaml 的 deepscaler 覆盖、ckpt pruner、内存防护段——全部已投产) |
| ~~zero_rm.py~~ | 已由 rm_hub 的 `rm_type == "zero"` 分支替代(commit 4b164e13) |
| `scripts/run-deepseek-v2-opd-smoke.sh`(已存在) | 同步加 flag,自蒸馏冒烟用它跑 |

**验收**:32 卡真实拓扑自蒸馏一步(`--opd-teacher-load` 指向学生自身 ckpt):
`opd_full_kl < 1e-3`、`grad_norm < 1e-2`;K∈{64,128,512} 扫描 `opd_full_kl` 单调收敛且
128→512 变化 <5%;显存峰值不超档1基线 +2GB;**对照实验**:同 checkpoint 各跑 N 步档1
(采样 token)vs 档2(top-K),比较 `eval/aime` 与 KL 下降斜率——这是 V4 论点
(全词表信号收敛更快)的直接检验。
**注**:v1 的"teacher ckpt 需 Megatron 格式"风险已消除——档1已用 HF 目录经 bridge 直载
teacher 投产(arguments.py:1696 仅 logger.info)。

### M5 —(可选,后置)多 teacher 与 sglang 路线

- **5a 多 teacher(MOPD,采纳 ms-swift 硬路由设计)**:`--opd-teacher-load` 接受 JSON
  `[{path, tags}]` + `--opd-teacher-tag-key`(默认 `dataset`,可指向样本 metadata 列);
  每样本按 tag 路由到**恰好一个** teacher(校验:多 teacher 时 tags 非空且不重叠、匹配失败
  fail-fast——校验规则照抄 `parse_teacher_model_server`);`weights_backuper` 注册
  teacher_0/1/...,每个 teacher 只对路由到它的样本位置 record top-K(其余位置掩零)。
  测试模板:ms-swift `tests/utils/test_multi_teacher.py`(405 行)。加权 KL 混合
  (V4 式29 的 wᵢ)留作 5a+,硬路由是 wᵢ∈{0,1} 的特例。单 teacher 需求下不做。
- **5b sglang top-K teacher**:teacher 引擎用 `top_logprobs_num` 直接吐 top-K,免训练侧
  teacher 常驻;代价是占 rollout 容量 + 需改 `sglang_rollout.py` 核心代码。仅当 Megatron
  teacher pass 时间(预计与 ref pass 同量级,每步 +4~5 分钟)不可接受时再评估。

---

## 3. 新增参数一览

| 参数 | 默认 | 说明 |
|---|---|---|
| `--opd-full-vocab-kl` | off | 启用 top-K 词表 KL loss;开启时 OPD 不再注入 advantage |
| `--opd-teacher-topk` | 128 | teacher 分布压缩宽度;=vocab 时精确(仅调试) |
| `--opd-distill-beta` | 1 | 1=反向 KL(V4 式29)/ 0=正向 / 中间=JSD(ms-swift 同款旋钮) |
| `--opd-distill-temperature` | 1 | 蒸馏温度,两侧 logits 同除 |
| `--opd-kl-coef`(复用) | — | KL loss 权重 λ |
| (`--opd-teacher-tail-mode`) | — | 移入 M1b(精确分解对照)一并实现,M1a renorm 不需要 |

## 4. 测试矩阵

| 层级 | 内容 | 里程碑 |
|---|---|---|
| 单元 | KL 对拍(K=V 精确/K=128 近似)、gradcheck、TP 一致性、尾部边界 | M1 |
| 单元 | K=1 退化 == 旧 teacher_log_probs;top-K 质量守恒;CP 2-D 切片 | M2 |
| 回归 | flag 关闭时全路径 bit-exact | M3 |
| 端到端 | 自蒸馏 KL≈0 & grad_norm≈0;K 扫描收敛;显存峰值 | M4 |

## 5. 风险清单(v2)

1. ~~CP 切片对齐~~——已通过 record/pop 微批传递整体移除(决策 A v2);残余风险是
   "teacher pass 与训练前向微批顺序不一致",由逐微批 shape 断言 + K=1 回归兜底;
2. ~~teacher ckpt 格式~~——已消除(档1投产验证 bridge 直载);
3. **renorm 的系统性低估**——top-K 内重归一忽略尾部质量,KL 偏小;监控
   `opd_teacher_topk_mass`(<0.95 的位置占比),K 扫描(M4)定量;必要时 M1b 精确分解;
4. **单测无 32 卡拓扑**——gloo 多进程小词表模拟,真实拓扑靠 M4 自蒸馏兜底;
5. **bf16 数值**——gather 后 KL 计算统一 fp32;全 -inf 位置丢弃保护;
6. **teacher pass 时间**——top-K 增量 ~10-20%(all_gather K 候选 + logsumexp);瓶颈时
   考虑与 ref pass 合并(纯 OPD 下 ref 可关)或 M5b。

## 6. 时间估算(v2)

M1a 0.5 天 → M2 1 天 → M3 0.5 天 → M4 0.5 天(脚本已在,只做自蒸馏冒烟 + K 扫描),
合计 **2.5~3 天**(不含 M1b 对照 +1 天、M5)。
