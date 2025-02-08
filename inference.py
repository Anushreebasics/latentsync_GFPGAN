# inference.py
# Copyright (c) 2024 Bytedance Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# (License text omitted for brevity)

import argparse
import cv2
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from LatentSync.latentsync.models.unet import UNet3DConditionModel
from LatentSync.latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from LatentSync.latentsync.whisper.audio2feature import Audio2Feature

# Import GFPGANer for super-resolution
from gfpgan import GFPGANer

def apply_superres_if_needed(ref_frame, gen_frame, gfpgan_model):
    """
    Compare the resolution of the generated frame (gen_frame) with a reference frame (ref_frame).
    If the generated frame is lower in either dimension, apply GFPGAN to upscale it.
    """
    ref_h, ref_w = ref_frame.shape[:2]
    gen_h, gen_w = gen_frame.shape[:2]

    if gen_h < ref_h or gen_w < ref_w:
        print("Generated frame resolution is lower than reference. Applying GFPGAN super-resolution...")
        # GFPGAN expects a BGR image and returns a restored image along with some additional info.
        restored_img, _ = gfpgan_model.enhance(
            gen_frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return restored_img
    else:
        return gen_frame

def save_video(frames, output_path, fps=30.0):
    """
    Save a list of frames as a video using OpenCV.
    """
    if not frames:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def main(config, args):
    # Determine appropriate torch dtype based on GPU capability.
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    # Initialize the scheduler.
    scheduler = DDIMScheduler.from_pretrained("configs")

    # Select the appropriate Whisper model based on cross_attention_dim.
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    # Initialize the audio encoder.
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    # Load the VAE.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # Load the UNet model.
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    # Enable memory-efficient attention if available.
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # Create the Lipsync pipeline.
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # Set the random seed.
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    print(f"Initial seed: {torch.initial_seed()}")

    # Run the pipeline to generate video frames.
    # (Assumption: The pipeline returns a list of frames as NumPy arrays.)
    output_frames = pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )

    # Read the first frame from the input video as a resolution reference.
    cap = cv2.VideoCapture(args.video_path)
    ret, ref_frame = cap.read()
    cap.release()
    if not ret:
        print("Warning: Could not read the input video; using first generated frame as reference.")
        ref_frame = output_frames[0]

    # Initialize the GFPGAN model if super-resolution is requested.
    if args.superres == "GFPGAN":
        # Update the model_path below if necessary.
        gfpgan_model = GFPGANer(
            model_path="GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
            upscale=2,            # Upscaling factor; adjust if needed
            arch="clean",         # Architecture type: "clean" for GFPGANv1.3
            channel_multiplier=2, # Channel multiplier; default is 2
            bg_upsampler=None
        )
    else:
        raise ValueError("Unsupported super-resolution model! Only GFPGAN is implemented in this example.")

    # Process each generated frame: if its resolution is lower than the reference, apply GFPGAN enhancement.
    processed_frames = []
    for idx, frame in enumerate(output_frames):
        print(f"Processing frame {idx+1}/{len(output_frames)}...")
        enhanced_frame = apply_superres_if_needed(ref_frame, frame, gfpgan_model)
        processed_frames.append(enhanced_frame)

    # Save the final processed video.
    save_video(processed_frames, args.video_out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], required=True,
                        help="Super-resolution model to use (currently only GFPGAN is implemented)")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)
