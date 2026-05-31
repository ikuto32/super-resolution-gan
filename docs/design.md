# super-resolution-gan Design Document

## 1. Project Summary

`super-resolution-gan` is a PyTorch-based research codebase for a resolution-agnostic super-resolution GAN built on the design principles of R3GAN.

The project targets conditional image super-resolution where a generator `G` maps a low-resolution image `x_A` to a high-resolution image `x_hat_B`, while a discriminator `D` evaluates real and generated images at arbitrary spatial resolutions.

The generator is designed as a progressive image expansion model. Instead of directly mapping `A → B` in a single pass, it starts from a spatially averaged `1x1` image and incrementally expands the image until it reaches the target resolution. The training objective combines adversarial R3GAN-style losses, multi-scale reconstruction losses, low-resolution consistency losses, perceptual losses, and an optional diffusion-style denoising auxiliary objective.

---

## 2. Core Objectives

### 2.1 Primary Goals

1. Implement a conditional super-resolution GAN using R3GAN-style adversarial training.
2. Support a discriminator that accepts arbitrary image resolutions.
3. Implement a progressive generator that expands images from `1x1` to target resolution.
4. Support multi-scale supervision using image pyramids.
5. Add diffusion-style denoising losses as auxiliary regularization, not as the primary sampler.
6. Keep the codebase modular, testable, and suitable for research iteration.

### 2.2 Non-Goals for the Initial Version

The initial implementation should not attempt to solve all real-world super-resolution cases.

The following are explicitly out of scope for the MVP:

* Full DDPM or latent diffusion sampling.
* Complex Real-ESRGAN-style degradation pipelines.
* Video super-resolution.
* Text-conditioned generation.
* Distributed training across multiple nodes.
* Production inference service.

---

## 3. High-Level Model Definition

The training data consists of high-resolution target images:

```text
y_B: high-resolution ground-truth image
```

A degradation pipeline produces the low-resolution input:

```text
x_A = degrade(y_B)
```

The generator predicts:

```text
x_hat_B = G(x_A, target_size=B)
```

The discriminator receives conditional image pairs:

```text
D(y_B, x_A)       # real pair
D(x_hat_B, x_A)   # fake pair
```

The key constraint is:

```text
downsample(x_hat_B) ≈ x_A
```

This constraint prevents the generator from hallucinating high-frequency detail that is visually plausible but inconsistent with the low-resolution input.

---

## 4. Repository Structure

The repository should use a clear separation between models, losses, datasets, training logic, configuration, and experiments.

```text
super-resolution-gan/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── sr_64_to_256.yaml
│   ├── sr_128_to_512.yaml
│   └── ablations/
│       ├── no_diffusion_loss.yaml
│       ├── no_multiscale_loss.yaml
│       └── unconditional_d.yaml
├── datasets/
│   ├── __init__.py
│   ├── image_pair_dataset.py
│   ├── degradation.py
│   ├── pyramid.py
│   ├── transforms.py
│   └── samplers.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── condition_encoder.py
│   │   ├── progressive_blocks.py
│   │   ├── diffusion_head.py
│   │   ├── normalization.py
│   │   └── common.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── r3gan.py
│   │   ├── reconstruction.py
│   │   ├── perceptual.py
│   │   ├── consistency.py
│   │   ├── diffusion.py
│   │   └── total.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   ├── checkpointing.py
│   │   ├── ema.py
│   │   ├── logging.py
│   │   └── validation.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   ├── lpips.py
│   │   ├── fid.py
│   │   └── consistency.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   ├── tiled_inference.py
│   │   └── export.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── distributed.py
│       ├── image_io.py
│       ├── random.py
│       └── tensor_ops.py
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── infer.py
│   ├── build_dataset.py
│   └── profile_model.py
├── tests/
│   ├── test_generator.py
│   ├── test_discriminator.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_training_step.py
├── notebooks/
│   └── inspection.ipynb
└── docs/
    ├── design.md
    ├── losses.md
    ├── training.md
    └── experiments.md
```

---

## 5. Development Environment

The project should use `uv` for Python environment management.

### 5.1 Initial Setup

```bash
uv init super-resolution-gan
cd super-resolution-gan
```

### 5.2 Suggested Dependencies

```bash
uv add torch torchvision torchaudio
uv add numpy pillow opencv-python tqdm pyyaml einops
uv add matplotlib tensorboard
uv add pytest ruff mypy
```

Optional research dependencies:

```bash
uv add lpips torchmetrics scipy
```

For CUDA-specific PyTorch installation, follow the PyTorch installation matrix and pin the correct wheel index in project documentation.

### 5.3 Running Training

```bash
uv run python scripts/train.py --config configs/sr_64_to_256.yaml
```

### 5.4 Running Tests

```bash
uv run pytest tests
```

### 5.5 Code Formatting

```bash
uv run ruff check .
uv run ruff format .
```

---

## 6. Dataset Design

### 6.1 Dataset Contract

The dataset should return both low-resolution and high-resolution images.

Each batch item should contain:

```text
{
    "lr": Tensor[C, A_H, A_W],
    "hr": Tensor[C, B_H, B_W],
    "hr_pyramid": dict[int, Tensor[C, H_s, W_s]],
    "meta": {
        "path": str,
        "scale": float,
        "original_size": tuple[int, int],
        "target_size": tuple[int, int]
    }
}
```

### 6.2 `datasets/image_pair_dataset.py`

Responsible for:

* Loading high-resolution images.
* Applying random crop.
* Generating low-resolution inputs through degradation.
* Returning image pyramids for multi-scale supervision.
* Supporting deterministic validation mode.

Core class:

```text
ImagePairDataset
```

Primary responsibilities:

```text
- Load HR image from disk.
- Convert image to RGB.
- Apply HR crop.
- Generate LR image via degradation pipeline.
- Build HR pyramid.
- Return tensors normalized to the configured range.
```

Recommended tensor range:

```text
[-1, 1]
```

### 6.3 `datasets/degradation.py`

Responsible for producing `x_A` from `y_B`.

Initial MVP degradation pipeline:

```text
1. Bicubic downsampling
2. Optional Gaussian blur
3. Optional Gaussian noise
4. Optional JPEG compression
5. Clamp to valid image range
```

The degradation pipeline should be configurable.

Example configuration:

```yaml
degradation:
  downsample:
    method: bicubic
    scale: 4
  blur:
    enabled: true
    sigma_min: 0.1
    sigma_max: 1.2
  noise:
    enabled: true
    std_min: 0.0
    std_max: 0.03
  jpeg:
    enabled: false
    quality_min: 70
    quality_max: 100
```

### 6.4 `datasets/pyramid.py`

Responsible for building multi-scale image pyramids.

Given a target HR image `y_B`, generate:

```text
y_1, y_2, y_4, y_8, ..., y_B
```

The pyramid should support:

* Power-of-two scales.
* Arbitrary final target size.
* Area or bicubic downsampling.
* Optional anti-aliasing.

For training stability, the smallest scale should be `1x1`:

```text
y_1x1 = spatial_mean(y_B)
```

---

## 7. Generator Design

### 7.1 Module

```text
src/models/generator.py
```

Primary class:

```text
ProgressiveSRGenerator
```

### 7.2 Generator Input

```text
x_A: low-resolution condition image
target_size: tuple[int, int]
noise: optional stochastic input
return_intermediates: bool
```

### 7.3 Generator Output

```text
{
    "image": x_hat_B,
    "pyramid": {
        1: x_hat_1,
        2: x_hat_2,
        4: x_hat_4,
        ...
    },
    "features": optional intermediate features
}
```

### 7.4 Generation Process

The generator starts from a `1x1` image initialized from the spatial mean of the low-resolution input:

```text
x_hat_1 = mean_spatial(x_A)
```

Then it progressively expands:

```text
x_hat_2   = G_1(x_hat_1, cond_2)
x_hat_4   = G_2(x_hat_2, cond_4)
x_hat_8   = G_3(x_hat_4, cond_8)
...
x_hat_B   = G_n(x_hat_prev, cond_B)
```

Each stage should follow a residual update pattern:

```text
x_hat_next = upsample(x_hat_current) + residual_prediction
```

This keeps the model biased toward low-frequency consistency and lets each block focus on adding missing detail.

### 7.5 Generator Submodules

#### 7.5.1 `ConditionEncoder`

File:

```text
src/models/condition_encoder.py
```

Responsible for encoding the low-resolution input `x_A`.

Outputs multi-scale condition features:

```text
{
    2: cond_2,
    4: cond_4,
    8: cond_8,
    ...
    B: cond_B
}
```

The condition encoder should support resizing or feature interpolation so that conditioning features can be injected at arbitrary generator scales.

#### 7.5.2 `ProgressiveUpsampleBlock`

File:

```text
src/models/progressive_blocks.py
```

Responsible for one scale transition:

```text
H x W → 2H x 2W
```

Each block should contain:

```text
- interpolation upsample
- convolutional residual block
- condition injection
- optional time/noise embedding injection
- RGB residual prediction
```

Recommended initial block design:

```text
input image/features
 ↓
upsample
 ↓
conv 3x3
 ↓
activation
 ↓
condition injection
 ↓
residual conv block
 ↓
to_rgb
 ↓
residual add
```

#### 7.5.3 Condition Injection

The initial implementation should use FiLM-style conditioning:

```text
h = gamma(cond) * h + beta(cond)
```

Where `gamma` and `beta` are predicted from condition features.

For the MVP, a simpler concatenation-based implementation is also acceptable:

```text
h = concat(h, resize(cond))
```

FiLM should become the preferred default after baseline validation.

---

## 8. Discriminator Design

### 8.1 Module

```text
src/models/discriminator.py
```

Primary class:

```text
ResolutionAgnosticDiscriminator
```

### 8.2 Discriminator Input

The discriminator is conditional.

```text
image: real or generated high-resolution image
condition: low-resolution input image
```

Before concatenation, the condition image is resized to match the candidate image resolution:

```text
condition_resized = resize(x_A, size=image.shape[-2:])
d_input = concat(image, condition_resized)
```

### 8.3 Discriminator Output

The discriminator should produce both patch logits and a scalar score.

```text
{
    "score": Tensor[B],
    "patch_logits": Tensor[B, 1, H_d, W_d],
    "features": optional intermediate features
}
```

The scalar score is produced using spatial aggregation:

```text
score = mean(patch_logits, dim=[2, 3])
```

No fixed-size flattening or fully connected layer should be used.

### 8.4 Resolution-Agnostic Requirements

The discriminator must satisfy the following constraints:

```text
- Fully convolutional architecture.
- No fixed spatial input size.
- No flatten operation that depends on H or W.
- Global average aggregation for scalar score.
- Same weights for all supported resolutions.
```

The discriminator should support images such as:

```text
64x64
96x96
128x128
192x192
256x256
512x512
```

The architecture must not assume power-of-two resolutions, although the generator may internally prefer power-of-two growth stages.

### 8.5 Multi-Scale Discrimination

The discriminator should be reusable across multiple scales:

```text
D(y_s, x_A)
D(x_hat_s, x_A)
```

This allows adversarial supervision at intermediate generator outputs.

In the MVP, adversarial loss may be applied only at the final scale. Multi-scale adversarial supervision should be introduced after the base training loop is stable.

---

## 9. Diffusion-Style Auxiliary Design

### 9.1 Module

```text
src/models/diffusion_head.py
src/losses/diffusion.py
```

The diffusion component should be auxiliary.

It should not replace the progressive generator with a full diffusion sampler in the initial version.

### 9.2 Training Objective

At selected scales, add noise to the ground-truth pyramid image:

```text
y_s_noisy = alpha_t * y_s + sigma_t * epsilon
```

The model predicts either:

```text
epsilon_hat
```

or:

```text
y_s_clean_hat
```

The initial version should predict noise:

```text
L_diff = || epsilon_hat - epsilon ||_2^2
```

### 9.3 Integration Strategy

The diffusion head should attach to intermediate generator features.

```text
feature_s → diffusion_head_s → epsilon_hat_s
```

This keeps the generator architecture clean and allows the diffusion objective to be enabled or disabled through configuration.

### 9.4 Recommended Initial Usage

The diffusion loss should be introduced only after the GAN + reconstruction baseline is stable.

Initial weight:

```text
lambda_diff = 0.01
```

---

## 10. Loss Design

### 10.1 Module Layout

```text
src/losses/
├── r3gan.py
├── reconstruction.py
├── perceptual.py
├── consistency.py
├── diffusion.py
└── total.py
```

### 10.2 Generator Loss

The generator objective:

```text
L_G =
    lambda_adv  * L_adv_G
  + lambda_pix  * L_pixel
  + lambda_ms   * L_multiscale
  + lambda_perc * L_perceptual
  + lambda_cons * L_lr_consistency
  + lambda_diff * L_diffusion
```

### 10.3 Discriminator Loss

The discriminator objective:

```text
L_D =
    L_adv_D
  + lambda_r1 * R1
  + lambda_r2 * R2
```

### 10.4 `src/losses/r3gan.py`

Responsible for adversarial losses.

Expected functions:

```text
discriminator_loss(real_scores, fake_scores, real_images, fake_images)
generator_loss(real_scores, fake_scores)
r1_regularization(real_scores, real_images)
r2_regularization(fake_scores, fake_images)
```

The implementation should keep the exact adversarial formulation isolated so that alternative GAN losses can be added later.

### 10.5 `src/losses/reconstruction.py`

Responsible for pixel-space losses.

Supported losses:

```text
- L1
- Charbonnier
- MSE
```

Default:

```text
L1
```

### 10.6 `src/losses/consistency.py`

Responsible for low-resolution consistency.

```text
L_consistency = || degrade_or_downsample(x_hat_B) - x_A ||_1
```

This loss is mandatory for super-resolution.

It ensures the generated image remains faithful to the input low-resolution observation.

### 10.7 `src/losses/perceptual.py`

Responsible for perceptual feature loss.

Initial implementation may use VGG-based perceptual loss.

The module should be optional because it introduces external model dependencies and additional GPU memory usage.

### 10.8 `src/losses/diffusion.py`

Responsible for diffusion-style denoising losses.

Expected functionality:

```text
- sample diffusion timestep t
- generate noisy target y_t
- compute epsilon target
- compute predicted epsilon loss
```

### 10.9 `src/losses/total.py`

Responsible for combining individual losses.

The output should be a structured dictionary:

```text
{
    "loss_total": Tensor,
    "loss_adv": Tensor,
    "loss_pixel": Tensor,
    "loss_multiscale": Tensor,
    "loss_perceptual": Tensor,
    "loss_consistency": Tensor,
    "loss_diffusion": Tensor
}
```

This structure should be used for logging and debugging.

---

## 11. Training Loop Design

### 11.1 Module

```text
src/training/trainer.py
```

Primary class:

```text
Trainer
```

### 11.2 Training Step

Each iteration should perform:

```text
1. Load batch:
   - lr
   - hr
   - hr_pyramid

2. Generator forward:
   - generated output
   - generated pyramid
   - optional diffusion predictions

3. Discriminator update (`training.n_critic` times):
   - compute D(real, lr)
   - compute D(fake.detach(), lr)
   - compute discriminator adversarial loss
   - compute regularization
   - update D

4. Generator update (once):
   - compute D(fake, lr)
   - compute adversarial generator loss
   - compute reconstruction losses
   - compute consistency loss
   - compute optional perceptual loss
   - compute optional diffusion loss
   - update G

5. EMA update:
   - update exponential moving average generator

6. Logging:
   - scalar losses
   - learning rates
   - generated samples
   - validation metrics
```

### 11.3 Progress Reporting

All long-running training, validation, dataset construction, and inference loops should use `tqdm`.

Progress bars should show:

```text
- current epoch
- current step
- total steps
- generator loss
- discriminator loss
- learning rate
```

Example progress description:

```text
epoch=3 step=1200 loss_g=1.284 loss_d=0.693 lr=2e-4
```

### 11.4 Checkpointing

Checkpoints should contain:

```text
{
    "step": int,
    "epoch": int,
    "generator": state_dict,
    "discriminator": state_dict,
    "generator_ema": state_dict,
    "optimizer_g": state_dict,
    "optimizer_d": state_dict,
    "scheduler_g": state_dict,
    "scheduler_d": state_dict,
    "config": dict,
    "rng_state": dict
}
```

Checkpoint files:

```text
checkpoints/
├── latest.pt
├── step_00010000.pt
├── step_00020000.pt
└── best_lpips.pt
```

Training must support resume:

```bash
uv run python scripts/train.py \
  --config configs/sr_64_to_256.yaml \
  --resume checkpoints/latest.pt
```

---

## 12. Configuration Design

The project should use YAML configuration files.

Example:

```yaml
project:
  name: super-resolution-gan
  output_dir: runs/sr_64_to_256

seed: 42

data:
  train_dir: data/train
  val_dir: data/val
  image_size_hr: [256, 256]
  image_size_lr: [64, 64]
  batch_size: 16
  num_workers: 8

degradation:
  downsample:
    method: bicubic
    scale: 4
  blur:
    enabled: true
    sigma_min: 0.1
    sigma_max: 1.2
  noise:
    enabled: true
    std_min: 0.0
    std_max: 0.03
  jpeg:
    enabled: false

model:
  generator:
    base_channels: 128
    max_channels: 512
    num_res_blocks_per_stage: 2
    condition_type: film
    start_from_mean_pixel: true
    return_intermediates: true

  discriminator:
    base_channels: 64
    max_channels: 512
    conditional: true
    input_condition_mode: concat
    patch_output: true
    global_aggregation: mean

loss:
  lambda_adv: 0.1
  lambda_pixel: 1.0
  lambda_multiscale: 0.5
  lambda_perceptual: 0.1
  lambda_consistency: 1.0
  lambda_diffusion: 0.0
  lambda_r1: 10.0
  lambda_r2: 0.0

training:
  epochs: 100
  mixed_precision: true
  grad_clip_norm: 1.0
  n_critic: 1
  log_every: 100
  validate_every: 1000
  save_every: 5000
  ema:
    enabled: true
    decay: 0.999

optimizer:
  generator:
    type: adamw
    lr: 0.0002
    betas: [0.0, 0.99]
    weight_decay: 0.0
  discriminator:
    type: adamw
    lr: 0.0002
    betas: [0.0, 0.99]
    weight_decay: 0.0
```

---

## 13. Inference Design

### 13.1 Module

```text
src/inference/predict.py
```

The inference API should accept:

```text
- input image path
- output image path
- target size or scale factor
- checkpoint path
- EMA flag
```

Example command:

```bash
uv run python scripts/infer.py \
  --checkpoint checkpoints/latest.pt \
  --input examples/lr.png \
  --output outputs/sr.png \
  --scale 4 \
  --ema
```

### 13.2 Tiled Inference

Large image inference should support tiling.

Module:

```text
src/inference/tiled_inference.py
```

Required features:

```text
- tile size
- overlap
- blending window
- padding
- automatic fallback for out-of-memory errors
```

Tiled inference is not required for the first MVP but should be planned from the beginning.

---

## 14. Metrics

### 14.1 Required Metrics

The validation loop should support:

```text
- PSNR
- SSIM
- LPIPS
- LR consistency error
```

### 14.2 Optional Metrics

Optional generative metrics:

```text
- FID
- KID
```

These should be implemented after a stable validation image export pipeline exists.

### 14.3 Validation Output

Each validation run should save:

```text
validation/
├── step_00010000/
│   ├── sample_000_lr.png
│   ├── sample_000_sr.png
│   ├── sample_000_hr.png
│   ├── sample_001_lr.png
│   ├── sample_001_sr.png
│   └── sample_001_hr.png
└── metrics.jsonl
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

Required test files:

```text
tests/test_generator.py
tests/test_discriminator.py
tests/test_losses.py
tests/test_dataset.py
tests/test_training_step.py
```

### 15.2 Generator Tests

Verify:

```text
- accepts arbitrary LR image sizes
- returns target-size image
- returns intermediate pyramid when requested
- starts from mean pixel when configured
- gradients flow through all progressive blocks
```

### 15.3 Discriminator Tests

Verify:

```text
- accepts 64x64, 96x96, 128x128, 256x256 inputs
- returns scalar score
- returns patch logits
- does not require fixed spatial dimensions
- works with conditional concatenation
```

### 15.4 Loss Tests

Verify:

```text
- each loss returns scalar tensor
- total loss returns structured dictionary
- R1 regularization supports autograd
- consistency loss downsamples generated image correctly
```

### 15.5 Training Step Test

A minimal training-step test should run on CPU with tiny tensors:

```text
batch size: 2
LR size: 8x8
HR size: 32x32
```

The test should confirm:

```text
- D update completes
- G update completes
- no NaN losses
- checkpoint save/load works
```

---

## 16. MVP Implementation Plan

### Phase 1: Conditional R3GAN Super-Resolution Baseline

Implement:

```text
- ImagePairDataset
- simple degradation pipeline
- direct G: LR → HR
- conditional resolution-agnostic D
- adversarial loss
- L1 loss
- LR consistency loss
- training loop
```

Target experiment:

```text
64x64 → 256x256
```

Success criteria:

```text
- training runs without divergence
- generated images match LR structure
- validation samples are visually plausible
- checkpoint resume works
```

### Phase 2: Progressive Generator

Replace direct generator with:

```text
1x1 → 2x2 → 4x4 → ... → target size
```

Add:

```text
- generated pyramid
- HR target pyramid
- multi-scale reconstruction loss
```

Success criteria:

```text
- intermediate outputs are meaningful
- final output quality is not worse than Phase 1
- multi-scale losses are logged correctly
```

### Phase 3: Multi-Scale Discriminator Training

Apply discriminator at selected pyramid scales:

```text
D(y_64, x_A)
D(x_hat_64, x_A)

D(y_128, x_A)
D(x_hat_128, x_A)

D(y_256, x_A)
D(x_hat_256, x_A)
```

Success criteria:

```text
- D accepts all selected resolutions
- adversarial loss remains stable
- high-frequency details improve
```

### Phase 4: Diffusion-Style Auxiliary Loss

Add:

```text
- diffusion noise schedule
- diffusion head
- epsilon prediction loss
- lambda_diff configuration
```

Success criteria:

```text
- diffusion loss decreases
- image texture quality improves or stabilizes
- no significant loss of LR consistency
```

### Phase 5: Arbitrary Resolution Support

Extend:

```text
- non-square image support
- arbitrary target size
- tiled inference
- validation on multiple resolutions
```

Success criteria:

```text
- D works on non-power-of-two sizes
- G can generate target sizes not limited to 256 or 512
- inference handles large images
```

---

## 17. Engineering Constraints

### 17.1 Determinism

The project should support deterministic runs where possible.

Required:

```text
- seed Python random
- seed NumPy
- seed PyTorch
- store RNG state in checkpoints
```

### 17.2 Mixed Precision

Mixed precision should be supported through configuration.

```yaml
training:
  mixed_precision: true
```

### 17.3 Device Handling

All scripts should support:

```text
- CPU fallback
- single CUDA device
- future DDP extension
```

### 17.4 Logging

Initial logging backend:

```text
TensorBoard
```

The logging module should not hardcode a specific external service.

Future options:

```text
- Weights & Biases
- MLflow
- custom JSONL logging
```

### 17.5 Failure Recovery

Long-running jobs must support:

```text
- periodic checkpoints
- latest checkpoint symlink or copy
- resume from checkpoint
- JSONL metrics append mode
```

---

## 18. Naming Conventions

### 18.1 Model Names

Use explicit names:

```text
ProgressiveSRGenerator
ResolutionAgnosticDiscriminator
ConditionEncoder
ProgressiveUpsampleBlock
DiffusionAuxiliaryHead
```

Avoid vague names:

```text
Net
Model
SRNet
GANModel
```

### 18.2 Tensor Naming

Use consistent tensor names:

```text
lr         # low-resolution input
hr         # high-resolution target
sr         # super-resolved output
fake       # generated image for D
real       # ground-truth image for D
cond       # condition image or features
```

### 18.3 Scale Naming

Use explicit scale keys:

```text
scale_1
scale_2
scale_4
scale_8
scale_16
...
```

or integer dictionary keys:

```text
pyramid[1]
pyramid[2]
pyramid[4]
...
```

Prefer integer keys internally.

---

## 19. Initial Research Questions

The codebase should be designed to answer the following research questions:

1. Does progressive generation from a `1x1` mean image improve stability over direct LR-to-HR generation?
2. Does a fully convolutional resolution-agnostic discriminator generalize across resolutions?
3. Does multi-scale adversarial supervision improve fine detail without increasing hallucination?
4. Does diffusion-style denoising regularization improve texture realism?
5. What is the trade-off between adversarial realism and low-resolution consistency?
6. Does FiLM conditioning outperform concatenation conditioning?
7. How sensitive is the model to degradation pipeline complexity?

---

## 20. Recommended Initial Defaults

```text
Task:               4x super-resolution
LR size:            64x64
HR size:            256x256
Batch size:         16
Generator channels: 128 base, 512 max
Discriminator:      fully convolutional conditional patch D
Losses:             adversarial + L1 + multiscale + consistency
Perceptual loss:    enabled after baseline
Diffusion loss:     disabled in MVP, enabled in Phase 4
Optimizer:          AdamW
EMA:                enabled
AMP:                enabled
Checkpointing:      every 5,000 steps
Validation:         every 1,000 steps
```

---

## 21. Final Architecture Summary

The target architecture is:

```text
Low-resolution image x_A
        │
        ├── ConditionEncoder ──────────────┐
        │                                  │
        ▼                                  │
spatial mean image, 1x1                    │
        │                                  │
        ▼                                  │
ProgressiveSRGenerator                     │
1x1 → 2x2 → 4x4 → ... → target size         │
        │                                  │
        ▼                                  │
Super-resolved image x_hat_B               │
        │                                  │
        ├── reconstruction losses           │
        ├── multi-scale losses              │
        ├── LR consistency loss             │
        ├── optional perceptual loss        │
        ├── optional diffusion loss         │
        │                                  │
        ▼                                  │
ResolutionAgnosticDiscriminator ◄──────────┘
        │
        ▼
R3GAN-style adversarial objective
```

The most important engineering principle is modularity.

The generator, discriminator, degradation pipeline, loss functions, and training loop must be independently testable. The first successful version should be simple, stable, and measurable. Research complexity should be added only after the baseline can train, resume, validate, and infer reliably.
