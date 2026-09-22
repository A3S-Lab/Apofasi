# Apofasi

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

一次前向传递完成类型化决策。不生成文本，也不需要再解析。

*Apófasi*（απόφαση）的意思是**决策**。宿主给出一份状态和一组类型化问题，Apofasi 返回一组类型化答案：`choice`（选择）、`score`（评分）或 `noul`（是/否）。JSON 契约与 TypeSafe [System One（Jev）](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 相同，宿主可以用同一套 schema 对接两个引擎。

架构：[`ARCHITECTURE.md`](ARCHITECTURE.md) ·
路线图：[`ROADMAP.md`](ROADMAP.md)

## 为什么需要它

决策不是补全。宿主已经知道问题类型、选项和评分刻度。生成式模型要先走一轮网络往返和逐 token 解码，再吐出一段文本，宿主还得自己解析。Apofasi 直接给类型化问题打分，并返回答案对象。

默认构建不引入任何机器学习框架。神经网络推理是可选功能。

## 使用

```rust
use a3s_apofasi::{Client, Criteria, DecisionKind, Question, State, SystemOneRequest};
use indexmap::IndexMap;
use serde_json::json;

fn main() -> a3s_apofasi::Result<()> {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("refunds")));
    opts.insert("other".into(), Some(json!("else")));
    let mut questions = IndexMap::new();
    questions.insert(
        "department".into(),
        Question::new(
            DecisionKind::Choice,
            json!("Which department?"),
            Some(Criteria::Choice(opts)),
        )?,
    );
    let res = Client::default().system_one(SystemOneRequest {
        model: None,
        state: State::Text("Please refund my invoice.".into()),
        questions,
    })?;
    let _ = res;
    Ok(())
}
```

`Client::default` 是词法引擎：纯 Rust，不加载权重。需要神经网络检查点时打开 `infer`。在 Apple Silicon 上，`metal` 和 `mlx` 选择 GPU 路径；`mlx` 是快路径，通过 `MLX_ROOT` 链接预编译的 `libmlx`。

```bash
cargo build --release
cargo build --release --features cli,metal,mlx --bin a3s-apofasi
```

## 相对 Jev

准确率和延迟来自未修改的 [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) `pilot-v1` 清单：seed `20260917`，AG News、Banking77、DAIR Emotion 各 100 条，没有失败。没有改该仓库，也没有改它的配置。Apofasi 使用的问题文本和 `label_000`… 选项键与该包里 Jev 适配器相同。概率由它的 `group_scores` 计分（ECE 10 箱，5% 错误预算）。检查点权重没有改。这次是 RTX 4090 上的 CUDA BF16 `english`。Jev **没有**在这台机器上运行。Jev 的数字是已发布的托管 API 结果，包含网络往返。速度列是本地 p50 相对已发布 Jev p50 快了多少倍。

| 数据集 | Apofasi 准确率 | Jev 准确率 | Apofasi p50 | Jev p50 | 速度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG News | 0.970 | 0.910 | 21.4 ms | 255.9 ms | 11.9× |
| DAIR Emotion | 0.430 | 0.480 | 22.4 ms | 236.3 ms | 10.5× |
| Banking77 | 0.710 | 0.870 | 143.1 ms | 246.4 ms | 1.7× |

AG News 准确率更高。Brier 是 0.098，对 Jev 的 0.146；ECE 是 0.147，对 Jev 的 0.064。DAIR Emotion 准确率低 0.050（Brier 0.975 对 0.846，ECE 0.434 对 0.351）。它只有 6 个标签，一次前向就放得下。带冒号模板的共享假设样板（`… emotion: anger`）在打包前会剥掉，让每个 `[MASK]` 紧挨区分性标签；没有冒号模板的开场白（AG News / Banking77）保持原样。Banking77 有 72 个标签。一次前向会把每个选项切成同一个 token 前缀（准确率 0.020，62% 的样本把真实标签的概率打成 0）。放不进 `head_max_len` 的选择题会交错拆成若干保留完整选项文本的组。拥挤的组还会把第二名（以及足够接近的第三名）送进下一轮。合成分布前两名接近时，只对近邻领先项（前 2 名，或第三名仍接近时前 3 名）做一次联合前向，避免干扰项再次引入 IIA 失败。准确率 0.710，Brier 0.493，ECE 0.226，真实标签概率为 0 的比例是 0。这高于已发布的 GLiNER2.5 准确率（0.610，p50 295.5 ms），仍比已发布的 Jev 低 0.160。多次前向使这一行是 1.7 倍，而不是单次前向的倍数。已发布的 GLiNER2.5 在另外两组上的准确率 / p50：AG News 0.700 / 44.9 ms，DAIR Emotion 0.440 / 43.3 ms。

这份试点之外，已发布的 Jev 延迟在 jev-benchmarks 的 4 标签和 6 标签任务上是 236–256 ms p50，在 [decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) 上是 264–276 ms p50。合在一起，公开的 Jev 单题是 **236–276 ms**。

## 先做门控，再决定要不要调用生成模型

宿主已经知道问题类型，否则就要花一整次补全去选标签、打分或判断是/否时，用 Apofasi。先跑类型化请求，再套宿主门控。`GatePolicy::default()` 只在选择题/评分的置信度至少 0.7，或 noul 极端度 `max(noul, 1 - noul)` 至少 0.7 时，才把答案留在 `Auto`。全部是 `Auto` 时，保留类型化答案，不再调用生成模型。任一题是 `Escalate` 时，最多再调用一次生成模型。它的文本是宿主侧证据，不会变成类型化 `Answer`。

```rust
let response = engine.decide(&request)?;
let gates = a3s_apofasi::gate_response(&response, &a3s_apofasi::GatePolicy::default());
if a3s_apofasi::any_escalate(&gates) {
    // 宿主侧最多一次生成。不要把这段散文解析成 Answer。
} else {
    // 使用 response.answers。这条路径的生成次数是 0。
}
```

`Client::default` 是词法引擎，不需要权重，适合测试。下面六个任务里，它的重叠分都低于 0.7，门控全部升级，生成调用省不下来。不要为了让这条路径变成 `Auto` 而降低阈值；那等于接受一个不确定的重叠分。

能跨过 0.7 的是神经网络 english 检查点。用 `infer` 加载（`NeuralEngine::load` 或 `load_with`）。Apple Silicon 上选 `metal` 或 `mlx`；设置了 `MLX_ROOT` 时 `mlx` 是快路径。把 `APOFASI_CHECKPOINT` 指到 bundle 根目录，英文状态文本由路由器选 `english`。

下表是这六个单题任务的一次配对测量，不是基准套件。词法时间是 debug 构建的中位数（预热 5 次，再测 50 次）。神经时间是 Apple M5 Max 上 release 的 Candle Metal：已发布的 english 检查点，预热 3 次，再取 20 次的 p50（`sorted[len/2]`）。生成模型列是每个任务一次托管的 DeepSeek V4.1 Flash 补全（输出上限 128 token，超时 90 秒），数字是提示 token 加补全 token。它没有和神经样本放在同一次运行里重测。这些行上神经编码器自己的用量是每次前向 40–57 个输入 token、8 个输出 token，不要和生成模型的 token 列混在一起。表里的标签是门控信号，不是在断言它们和生成模型一致。

| 任务 | 词法 | 神经 p50 | 神经门控 | 生成模型 |
| --- | --- | ---: | --- | --- |
| 退款分流 | 97 µs，升级（billing 0.11） | 16.1 ms | **自动**（billing 0.87） | 2552 ms，152/56 |
| 结账中断 | 81 µs，升级（noul 0.62） | 15.7 ms | **自动**（noul 0.85） | 2839 ms，144/115 |
| 密钥调试补丁 | 83 µs，升级（noul 0.62） | 16.1 ms | 升级（noul 0.42） | 2483 ms，152/93 |
| 搜索命令 | 87 µs，升级（只读 0.11） | 16.2 ms | 升级（只读 0.16） | 2301 ms，151/69 |
| 清构建并测试 | 89 µs，升级（改写 0.11） | 16.0 ms | 升级（改写 0.55） | 2573 ms，148/91 |
| 强制推送 | 78 µs，升级（score 1.00，置信度 0.00） | 15.1 ms | 升级（score 1.15，置信度 0.03） | 3495 ms，122/128 |

六个神经答案里有两个跨过了 0.7，跳过了补全：大约 5.4 秒和 467 个 token，每次大约 16 ms。另外四个仍然升级，神经前向只是在同一次补全之前多加大约 16 ms，不是节省。加速只发生在 `Auto` 上。词法前向短得多，但这组任务上它从不到 `Auto`，所以什么也没省下。

```bash
cargo run --release --features cli,metal --bin a3s-apofasi -- suite \
  --cases cases.json --checkpoint "$APOFASI_CHECKPOINT" \
  --device metal --warmup 3 --iters 20
```

下面七个用例是另一套本地测量。每个用例先预热 12 次，再计时 40 次。数字是 p50（上中位数：排序后取 `sorted[len/2]`）。它们没有可并排的已发布 Jev 准确率。七个用例都通过了记录这些样本时使用的答案检查。只有前两行和 Jev 公开数据里的「一次决策」形状相同。

| 用例 | 问题数 | Apofasi p50 | 相对 236–276 ms |
| --- | ---: | ---: | --- |
| 单条退款问题 | 1 | 7.02 ms | 33.6×–39.3× |
| 含糊的部门选择 | 1 | 7.32 ms | 32.2×–37.7× |
| 三种原语 | 3 | 8.77 ms | 26.9×–31.5× |
| 英文账单分诊 | 4 | 12.57 ms | 18.8×–22.0× |
| 中文账单分诊 | 4 | 5.04 ms | 46.8×–54.8× |
| 防护预设 | 5 | 12.76 ms | 18.5×–21.6× |
| 显式指定检查点 | 4 | 12.82 ms | 18.4×–21.5× |

## 检查点

神经网络运行需要下面这棵目录（`APOFASI_CHECKPOINT`）：

```text
checkpoint/
├── rl_agent_config.json
├── model.safetensors
├── encoder/config.json
└── tokenizer/tokenizer.json
```

一个包可以在同一根目录下嵌套 `english/`、`multilingual/` 和 `typed-decisions/`。路由器在前向之前选定检查点。详见 [`docs/publish-layout.md`](docs/publish-layout.md)。

## Crate

| 项 | 值 |
| --- | --- |
| 包名 | `a3s-apofasi` |
| 版本 | 0.1.1 |
| 仓库 | [A3S-Lab/Apofasi](https://github.com/A3S-Lab/Apofasi) |
| 许可证 | MIT |

## 许可证

MIT © A3S Lab
