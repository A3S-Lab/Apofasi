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

Apofasi 0.1.1 在 Apple Silicon 上用 MLX 前向路径测量。Jev **没有**在这台机器上运行。下面的 Jev 数字是已发布的托管 API 结果，包含网络往返。Apofasi 的数字是设备上预热后的时间。速度列是本地时间相对已发布 Jev p50 快了多少倍，不是同一台机器上的配对重测。

准确率来自未修改的 [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) `pilot-v1` 清单：AG News、Banking77、DAIR Emotion 各 100 条，没有失败。没有改该仓库，也没有改它的配置。Apofasi 使用的问题文本和 `label_000`… 选项键与该包里 Jev 适配器相同。概率由它的 `group_scores` 计分（ECE 10 箱，5% 错误预算）。检查点权重没有改。

| 数据集 | Apofasi 准确率 | Jev 准确率 | Apofasi p50 | Jev p50 | 速度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG News | 0.960 | 0.910 | 9.8 ms | 255.9 ms | 26.1× |
| DAIR Emotion | 0.420 | 0.480 | 13.1 ms | 236.3 ms | 18.0× |
| Banking77 | 0.560 | 0.870 | 76.1 ms | 246.4 ms | 3.2× |

AG News 准确率更高，并且快 26.1 倍。Brier 是 0.098，对 Jev 的 0.146；ECE 是 0.137，对 Jev 的 0.064。DAIR Emotion 准确率低 0.060（Brier 0.976 对 0.846，ECE 0.404 对 0.351），快 18.0 倍。它只有 6 个标签，一次前向就放得下。Banking77 有 72 个标签。一次前向会把每个选项切成同一个 token 前缀（准确率 0.020，62% 的样本把真实标签的概率打成 0）。放不进 `head_max_len` 的选择题现在会拆成若干放得进的组，再比较各组的胜出项。准确率 0.560，Brier 0.599，ECE 0.134，真实标签概率为 0 的比例是 0。这仍比已发布的 Jev 低 0.310，也略低于已发布的 GLiNER2.5（准确率 0.610，p50 295.5 ms）。多次前向使这一行是 3.2 倍，而不是单次前向的倍数。已发布的 GLiNER2.5 在另外两组上的准确率 / p50：AG News 0.700 / 44.9 ms，DAIR Emotion 0.440 / 43.3 ms。

这份试点之外，已发布的 Jev 延迟在 jev-benchmarks 的 4 标签和 6 标签任务上是 236–256 ms p50，在 [decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) 上是 264–276 ms p50。合在一起，公开的 Jev 单题是 **236–276 ms**。

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
