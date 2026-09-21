// Decision forward on Apple MLX.
//
// Linked against a prebuilt libmlx. Weights stay f32 so option logits match
// the Candle path. GeGLU uses erf GELU, the same activation as the encoder.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace {

using mlx::core::array;
using mlx::core::astype;
using mlx::core::eval;
using mlx::core::float32;
using mlx::core::int32;
using mlx::core::matmul;
using mlx::core::reshape;
using mlx::core::split;
using mlx::core::take;
using mlx::core::transpose;

void set_err(char* err, size_t n, const std::string& msg) {
  if (err == nullptr || n == 0) {
    return;
  }
  std::snprintf(err, n, "%s", msg.c_str());
}

array gelu_erf(const array& x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  return x * (0.5f + 0.5f * mlx::core::erf(x * kInvSqrt2));
}

array gelu_tanh(const array& x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return x * (0.5f + 0.5f * mlx::core::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

array affine(const array& x, const array& weight, const std::optional<array>& bias) {
  array y = matmul(x, transpose(weight));
  if (bias.has_value()) {
    y = y + *bias;
  }
  return y;
}

array layer_norm(const array& x, const array& weight, const array* bias, float eps) {
  std::optional<array> bias_arg = bias == nullptr ? std::nullopt : std::optional<array>(*bias);
  return mlx::core::fast::layer_norm(x, std::optional<array>(weight), bias_arg, eps);
}

array sdpa(const array& q, const array& k, const array& v, float scale, const std::optional<array>& mask) {
  return mlx::core::fast::scaled_dot_product_attention(
      q, k, v, scale, "", mask, std::nullopt, false);
}

struct EncLayer {
  std::optional<array> attn_norm;
  array wqkv{0.0f};
  array wo{0.0f};
  array mlp_norm{0.0f};
  array wi{0.0f};
  array wo_mlp{0.0f};
  bool local = false;
  float theta = 0.0f;
};

struct HeadLayer {
  array in_w{0.0f};
  array in_b{0.0f};
  array out_w{0.0f};
  array out_b{0.0f};
  array linear1_w{0.0f};
  array linear1_b{0.0f};
  array linear2_w{0.0f};
  array linear2_b{0.0f};
  array norm1_w{0.0f};
  array norm1_b{0.0f};
  array norm2_w{0.0f};
  array norm2_b{0.0f};
};

array attend(
    const array& x,
    const array& wqkv,
    const std::optional<array>& bqkv,
    const array& wo,
    const std::optional<array>& bo,
    int b,
    int seq,
    int n_heads,
    int head_dim,
    std::optional<float> rope_theta,
    const std::optional<array>& mask) {
  array qkv = reshape(affine(x, wqkv, bqkv), {b, seq, 3, n_heads, head_dim});
  auto parts = split(qkv, 3, 2);
  array q = transpose(reshape(parts[0], {b, seq, n_heads, head_dim}), {0, 2, 1, 3});
  array k = transpose(reshape(parts[1], {b, seq, n_heads, head_dim}), {0, 2, 1, 3});
  array v = transpose(reshape(parts[2], {b, seq, n_heads, head_dim}), {0, 2, 1, 3});
  if (rope_theta.has_value()) {
    // MLX RoPE treats axis -2 as the sequence length: (B, *, T, D).
    q = mlx::core::fast::rope(q, head_dim, false, *rope_theta, 1.0f, 0);
    k = mlx::core::fast::rope(k, head_dim, false, *rope_theta, 1.0f, 0);
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  array att = sdpa(q, k, v, scale, mask);
  att = reshape(transpose(att, {0, 2, 1, 3}), {b, seq, n_heads * head_dim});
  return affine(att, wo, bo);
}

array mask_array(const uint8_t* attn, int batch, int seq, std::optional<int> radius) {
  std::vector<float> data(static_cast<size_t>(batch) * seq * seq, 0.0f);
  for (int row = 0; row < batch; ++row) {
    for (int i = 0; i < seq; ++i) {
      for (int j = 0; j < seq; ++j) {
        bool blocked = attn[static_cast<size_t>(row * seq + j)] == 0;
        if (radius.has_value() && std::abs(i - j) > *radius) {
          blocked = true;
        }
        if (blocked) {
          data[static_cast<size_t>((row * seq + i) * seq + j)] = -1.0e9f;
        }
      }
    }
  }
  return array(data.begin(), {batch, 1, seq, seq});
}

class Net {
 public:
  int hidden = 0;
  int n_heads = 0;
  int head_dim = 0;
  int local_radius = 0;
  float eps = 1.0e-5f;
  array embed{0.0f};
  array embed_norm{0.0f};
  std::vector<EncLayer> layers;
  array final_norm{0.0f};
  array type_emb{0.0f};
  std::vector<HeadLayer> head;
  array scorer_norm_w{0.0f};
  array scorer_norm_b{0.0f};
  array scorer_fc1_w{0.0f};
  array scorer_fc1_b{0.0f};
  array scorer_fc2_w{0.0f};
  array scorer_fc2_b{0.0f};

  void forward(
      const int32_t* ids,
      const uint8_t* attn,
      int batch,
      int seq,
      const int32_t* markers,
      int n_markers,
      const uint8_t* qtypes,
      float* out) const {
    array id_t(ids, {batch, seq}, int32);
    array hidden = layer_norm(take(embed, id_t, 0), embed_norm, nullptr, eps);

    const bool dense = std::all_of(attn, attn + static_cast<size_t>(batch) * seq, [](uint8_t v) {
      return v == 1;
    });
    std::optional<array> global_mask;
    std::optional<array> local_mask;
    if (!dense) {
      global_mask = mask_array(attn, batch, seq, std::nullopt);
    }
    if (!(dense && seq <= local_radius + 1)) {
      local_mask = mask_array(attn, batch, seq, local_radius);
    }

    for (const EncLayer& layer : layers) {
      const std::optional<array>& mask = layer.local ? local_mask : global_mask;
      array attn_in = hidden;
      if (layer.attn_norm.has_value()) {
        attn_in = layer_norm(hidden, *layer.attn_norm, nullptr, eps);
      }
      array attended = attend(
          attn_in,
          layer.wqkv,
          std::nullopt,
          layer.wo,
          std::nullopt,
          batch,
          seq,
          n_heads,
          head_dim,
          layer.theta,
          mask);
      hidden = hidden + attended;
      array mlp_in = layer_norm(hidden, layer.mlp_norm, nullptr, eps);
      array mid = affine(mlp_in, layer.wi, std::nullopt);
      auto parts = split(mid, 2, -1);
      array gated = gelu_erf(parts[0]) * parts[1];
      hidden = hidden + affine(gated, layer.wo_mlp, std::nullopt);
    }
    hidden = layer_norm(hidden, final_norm, nullptr, eps);

    std::vector<int32_t> qtype_ids(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
      qtype_ids[static_cast<size_t>(i)] = static_cast<int32_t>(qtypes[i]);
    }
    array qtype(qtype_ids.data(), {batch}, int32);
    hidden = hidden + reshape(take(type_emb, qtype, 0), {batch, 1, this->hidden});

    for (const HeadLayer& layer : head) {
      array normed = layer_norm(hidden, layer.norm1_w, &layer.norm1_b, 1.0e-5f);
      array attended = attend(
          normed,
          layer.in_w,
          layer.in_b,
          layer.out_w,
          layer.out_b,
          batch,
          seq,
          n_heads,
          head_dim,
          std::nullopt,
          global_mask);
      hidden = hidden + attended;
      normed = layer_norm(hidden, layer.norm2_w, &layer.norm2_b, 1.0e-5f);
      array ff = gelu_tanh(affine(normed, layer.linear1_w, layer.linear1_b));
      hidden = hidden + affine(ff, layer.linear2_w, layer.linear2_b);
    }

    array idx(markers, {n_markers}, int32);
    array flat = reshape(hidden, {batch * seq, this->hidden});
    array gathered = take(flat, idx, 0);
    array scores = layer_norm(gathered, scorer_norm_w, &scorer_norm_b, 1.0e-5f);
    scores = gelu_tanh(affine(scores, scorer_fc1_w, scorer_fc1_b));
    scores = affine(scores, scorer_fc2_w, scorer_fc2_b);
    scores = reshape(scores, {n_markers});
    eval(scores);
    std::memcpy(out, scores.data<float>(), static_cast<size_t>(n_markers) * sizeof(float));
  }
};

array require(std::unordered_map<std::string, array>& weights, const std::string& key) {
  auto it = weights.find(key);
  if (it == weights.end()) {
    throw std::runtime_error("missing tensor " + key);
  }
  return astype(it->second, float32);
}

std::optional<array> optional_weight(
    std::unordered_map<std::string, array>& weights,
    const std::string& key) {
  auto it = weights.find(key);
  if (it == weights.end()) {
    return std::nullopt;
  }
  return astype(it->second, float32);
}

}  // namespace

struct ApofasiMlxConfig {
  int32_t hidden;
  int32_t n_layers;
  int32_t n_heads;
  int32_t global_every;
  int32_t local_attention;
  int32_t head_layers;
  float global_theta;
  float local_theta;
  float eps;
};

struct MlxNet {
  Net net;
};

extern "C" {

MlxNet* apofasi_mlx_load(const char* path, const ApofasiMlxConfig* cfg, char* err, size_t err_len) {
  try {
    if (path == nullptr || cfg == nullptr) {
      set_err(err, err_len, "null mlx load argument");
      return nullptr;
    }
    if (cfg->n_heads <= 0 || cfg->hidden % cfg->n_heads != 0 || cfg->global_every <= 0) {
      set_err(err, err_len, "invalid encoder config");
      return nullptr;
    }
    auto loaded = mlx::core::load_safetensors(std::string(path));
    auto& weights = loaded.first;
    auto net = std::make_unique<MlxNet>();
    Net& model = net->net;
    model.hidden = cfg->hidden;
    model.n_heads = cfg->n_heads;
    model.head_dim = cfg->hidden / cfg->n_heads;
    model.local_radius = cfg->local_attention / 2;
    model.eps = cfg->eps;
    model.embed = require(weights, "encoder.embeddings.tok_embeddings.weight");
    model.embed_norm = require(weights, "encoder.embeddings.norm.weight");
    model.layers.reserve(static_cast<size_t>(cfg->n_layers));
    for (int32_t i = 0; i < cfg->n_layers; ++i) {
      const std::string prefix = "encoder.layers." + std::to_string(i);
      const bool local = i % cfg->global_every != 0;
      EncLayer layer;
      layer.attn_norm = optional_weight(weights, prefix + ".attn_norm.weight");
      layer.wqkv = require(weights, prefix + ".attn.Wqkv.weight");
      layer.wo = require(weights, prefix + ".attn.Wo.weight");
      layer.mlp_norm = require(weights, prefix + ".mlp_norm.weight");
      layer.wi = require(weights, prefix + ".mlp.Wi.weight");
      layer.wo_mlp = require(weights, prefix + ".mlp.Wo.weight");
      layer.local = local;
      layer.theta = local ? cfg->local_theta : cfg->global_theta;
      model.layers.push_back(std::move(layer));
    }
    model.final_norm = require(weights, "encoder.final_norm.weight");
    model.type_emb = require(weights, "type_emb.weight");
    model.head.reserve(static_cast<size_t>(cfg->head_layers));
    for (int32_t i = 0; i < cfg->head_layers; ++i) {
      const std::string prefix = "head.layers." + std::to_string(i);
      HeadLayer layer;
      layer.in_w = require(weights, prefix + ".self_attn.in_proj_weight");
      layer.in_b = require(weights, prefix + ".self_attn.in_proj_bias");
      layer.out_w = require(weights, prefix + ".self_attn.out_proj.weight");
      layer.out_b = require(weights, prefix + ".self_attn.out_proj.bias");
      layer.linear1_w = require(weights, prefix + ".linear1.weight");
      layer.linear1_b = require(weights, prefix + ".linear1.bias");
      layer.linear2_w = require(weights, prefix + ".linear2.weight");
      layer.linear2_b = require(weights, prefix + ".linear2.bias");
      layer.norm1_w = require(weights, prefix + ".norm1.weight");
      layer.norm1_b = require(weights, prefix + ".norm1.bias");
      layer.norm2_w = require(weights, prefix + ".norm2.weight");
      layer.norm2_b = require(weights, prefix + ".norm2.bias");
      model.head.push_back(std::move(layer));
    }
    model.scorer_norm_w = require(weights, "scorer.0.weight");
    model.scorer_norm_b = require(weights, "scorer.0.bias");
    model.scorer_fc1_w = require(weights, "scorer.1.weight");
    model.scorer_fc1_b = require(weights, "scorer.1.bias");
    model.scorer_fc2_w = require(weights, "scorer.3.weight");
    model.scorer_fc2_b = require(weights, "scorer.3.bias");
    return net.release();
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    return nullptr;
  }
}

void apofasi_mlx_free(MlxNet* net) {
  delete net;
}

int apofasi_mlx_forward(
    MlxNet* net,
    const int32_t* ids,
    const uint8_t* attn,
    int32_t batch,
    int32_t seq,
    const int32_t* markers,
    int32_t n_markers,
    const uint8_t* qtypes,
    float* out,
    char* err,
    size_t err_len) {
  try {
    if (net == nullptr || ids == nullptr || attn == nullptr || markers == nullptr || qtypes == nullptr ||
        out == nullptr || batch <= 0 || seq <= 0 || n_markers <= 0) {
      set_err(err, err_len, "invalid mlx forward argument");
      return -1;
    }
    net->net.forward(ids, attn, batch, seq, markers, n_markers, qtypes, out);
    return 0;
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    return -1;
  }
}

}  // extern "C"
