# GLM-Image for text-to-image and image editing on 2× or 1× A800 80GB

## Summary

- Vendor: Z.ai 
- Model: `zai-org/GLM-Image`
- Task: Text-to-image (T2I) and image-to-image / editing (I2I)
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`zai-org/GLM-Image` with vLLM-Omni on **two 80 GB NVIDIA A800** GPUs (Ampere-class,
same default layout as the upstream **2×A100 80GB** example: Stage 0 AR on GPU 0,
Stage 1 diffusion on GPU 1) and validate the deployment with the existing
`examples/online_serving/glm_image` clients. For **one** A800 80 GB GPU, follow
the **1× A800 80GB** section below (custom stage YAML required).

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/glm_image.md`](../../docs/user_guide/examples/online_serving/glm_image.md)
- Related example under `examples/`:
  [`examples/online_serving/glm_image/README.md`](../../examples/online_serving/glm_image/README.md)
- Related issue or discussion:
  [#2888](https://github.com/vllm-project/vllm-omni/pull/2888)

## Hardware Support

This recipe documents **dual-GPU** and **single-GPU** CUDA layouts on A800 80 GB
for the same software stack. Add more platforms (for example ROCm / NPU) as
community validation lands.

## GPU

### 2× A800 80GB

#### Environment

These versions were taken from a working **editable** install: activate `vllm-omni/.venv` (or your equivalent), then align `pip` / Git with the rows below when reproducing this recipe.

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA stack with **two** A800 80 GB GPUs visible (set `CUDA_VISIBLE_DEVICES` on your host if needed)
- vLLM: **0.19.0**
- vLLM-Omni: **0.19.0rc2.dev138+g38d5f2d53** (editable install from this repo; Git **`38d5f2d5`**, `git describe` ≈ **`v0.19.0rc1-138-g38d5f2d5`**)
- Transformers: **5.5.4** (same `.venv` as above; required so `glm_image` configs load for Stage 0)

#### Command

Start the server from the repository root:

```bash
vllm serve zai-org/GLM-Image --omni --port 8091
```

To use the bundled stage config explicitly (same default as above):

```bash
vllm serve zai-org/GLM-Image \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/glm_image.yaml
```

#### Verification

Run one of the existing example clients after the server is ready:

```bash
python examples/online_serving/glm_image/openai_chat_client.py \
  --prompt "A cute cat sitting on a window sill" \
  --output glm_image_output.png \
  --server http://localhost:8091
```
After the command finishes, check for the output files:

```bash
ls glm_image_output.png
```

#### Notes

- Memory usage: Roughly **~18 GiB + KV** on Stage 0 (AR) and **~20 GiB** on Stage 1 (DiT+VAE) per the user guide; two 80 GB cards match the default split.
- Key flags: `--omni` is required; `--stage-configs-path` is optional unless you use a custom YAML (for example single-GPU). 
- Keep **Transformers ≥ 5.5.1** (this recipe used **5.5.4**) so `glm_image` configs resolve; otherwise Stage 0 can fail at `ModelConfig` validation.
- Known limitations: This starter recipe follows the dual-GPU online path documented under `examples/online_serving/glm_image`. The first request may be slower due to warmup.
- Generation time: ~62s end-to-end on 2× A800 80GB (50 inference steps, 1024×1024, chat completions client).

### 1× A800 80GB

Default `glm_image.yaml` pins Stage 0 to GPU **0** and Stage 1 to GPU **1**.
On a single card, both stages must use the **same** device id.

#### Environment

Same software stack as **2× A800 80GB** (Python **3.12**, vLLM **0.19.0**,
vLLM-Omni **0.19.0rc2.dev138+g38d5f2d53**, Transformers **5.5.4**), but only **one**
A800 80 GB GPU visible (often `CUDA_VISIBLE_DEVICES=0`).

#### Command

1. Copy the stock stage file and point **Stage 1** at the same GPU as Stage 0:

```bash
cp vllm_omni/model_executor/stage_configs/glm_image.yaml \
  vllm_omni/model_executor/stage_configs/glm_image_single_gpu.yaml
```

In `glm_image_single_gpu.yaml`, Stage 0 already has `runtime.devices: "0"`.
Under **Stage 1** (`stage_id: 1`), change only the device line from `"1"` to `"0"`:

```yaml
  - stage_id: 1
    stage_type: diffusion
    runtime:
      process: true
      devices: "0"   # was "1" in the default dual-GPU file
      requires_multimodal_data: true
```

2. Start the server with your file:

```bash
vllm serve zai-org/GLM-Image \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/glm_image_single_gpu.yaml
```

If you hit **OOM**, lower Stage 0 `engine_args.gpu_memory_utilization` in the same
YAML (for example from `0.6` to `0.5` or `0.45`) and retry; see the
[GLM-Image user guide FAQ](../../docs/user_guide/examples/online_serving/glm_image.md#faq).

#### Verification

Same commands as **2× A800 80GB** (Python client and `curl` smoke test); only
the server startup line changes because of `--stage-configs-path`.

#### Notes

- Memory usage: AR and diffusion **time-share** one 80 GB GPU; peak usage is
  higher than the dual-GPU split. The user-guide ballpark (~48 GiB + KV for AR,
  ~22 GiB for DiT+VAE) ~72G for inference in total
- Key flags: **`--stage-configs-path`** is **required** for single-GPU; the
  default bundle still targets two GPUs.
- Keep Transformers **≥ 5.5.1** (here
  **5.5.4**) for `glm_image` support.
- Known limitations: Stages no longer run on separate devices in parallel;
  throughput differs from the 2× recipe. 
- Generation time: ~62s end-to-end on 2× A800 80GB (50 inference steps, 1024×1024, chat completions client).


