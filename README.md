# LatentSync with GFPGAN Super-Resolution

This repository demonstrates how to integrate GFPGAN super-resolution into the [LatentSync](https://github.com/bytedance/LatentSync) lipsync pipeline. The integration automatically applies GFPGAN enhancement to the generated video frames only when their resolution is lower than the original input video frames.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
  - [1. Clone Repositories & Install Dependencies](#1-clone-repositories--install-dependencies)
  - [2. Download GFPGAN Pre-trained Weights](#2-download-gfpgan-pre-trained-weights)
  - [3. Upgrade JAX (if needed)](#3-upgrade-jax-if-needed)
- [Modified Inference Pipeline](#modified-inference-pipeline)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Features

- **LatentSync Lipsync Pipeline:** Generate lipsynced video frames using LatentSync.
- **Conditional GFPGAN Enhancement:** Compares the generated frame’s resolution with the original input; if lower, applies GFPGAN super-resolution.
- **Google Colab Ready:** All instructions and commands are provided for running in a Colab notebook.

