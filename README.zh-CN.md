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

## 相对 Jev 的延迟

Apofasi 0.1.0 在 Apple Silicon 上用 MLX 前向路径测量。检查点保持常驻。每个用例先预热 12 次，再计时 40 次。数字是 p50（上中位数：排序后取 `sorted[len/2]`）。

Jev **没有**在这台机器上运行。下面的区间是公开的托管 API 上、一次类型化决策的 p50：

- [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)：Jev 1.13.0 在 4 标签和 6 标签任务上的 p50 为 236–256 ms，72 标签为 246 ms。延迟包含基准客户端到服务的网络往返。
- [decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)：Jev 的 p50 为 264–276 ms，从 2 个选项到 255 个选项基本持平。

合在一起，公开的 Jev 单题 p50 是 **236–276 ms**。Apofasi 的数字是设备上的前向时间。倍数表示本地调用少花了多少时间，不是把那两份研究的题目再跑一遍。

| 用例 | 问题数 | Apofasi p50 | 相对 236–276 ms |
| --- | ---: | ---: | --- |
| 单条退款问题 | 1 | 7.02 ms | 33.6×–39.3× |
| 含糊的部门选择 | 1 | 7.32 ms | 32.2×–37.7× |
| 三种原语 | 3 | 8.77 ms | 26.9×–31.5× |
| 英文账单分诊 | 4 | 12.57 ms | 18.8×–22.0× |
| 中文账单分诊 | 4 | 5.04 ms | 46.8×–54.8× |
| 防护预设 | 5 | 12.76 ms | 18.5×–21.6× |
| 显式指定检查点 | 4 | 12.82 ms | 18.4×–21.5× |

只有前两行和 Jev 公开数据里的「一次决策」形状相同。其余行一次调用里的问题更多，总耗时仍然低于那个单题区间。七个用例都通过了记录这些样本时使用的同一套答案检查。

这张表不是准确率对比。Jev 公开的标签准确率（jev-benchmarks 中 AG News 0.910、Banking77 0.870、DAIR Emotion 0.480）是在那些数据集上测的。Apofasi 0.1.0 没有在上面重测。

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
| 版本 | 0.1.0 |
| 仓库 | [A3S-Lab/Apofasi](https://github.com/A3S-Lab/Apofasi) |
| 许可证 | MIT |

## 许可证

MIT © A3S Lab
