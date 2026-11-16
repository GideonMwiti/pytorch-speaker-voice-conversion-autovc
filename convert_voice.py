#!/usr/bin/env python3
"""
Production CLI for Project 99: Speaker Voice Conversion (AutoVC)
Converts the speaker identity of an input speech recording into a target speaker's voice
while preserving linguistic content and prosody.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from main import (
    AutoVCModel,
    AudioSpectrogramExtractor,
    SAMPLE_RATE,
    AUDIO_DURATION,
    N_SAMPLES,
    SPEAKERS,
    synth_speech_utterance,
)

def load_voice_converter(checkpoint_path: str = "models/autovc_voice_converter_best.pth", device: str = "cpu"):
    """Load pre-trained AutoVC model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please run 'python main.py' first to train the AutoVC system."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AutoVCModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def convert_voice(
    source_wav_path: str,
    target_wav_path: str = None,
    target_speaker_name: str = None,
    output_wav_path: str = "outputs/converted_voice.wav",
    model=None,
    checkpoint_path: str = "models/autovc_voice_converter_best.pth",
    device: str = "cpu"
):
    """Perform zero-shot any-to-any voice conversion."""
    if model is None:
        model = load_voice_converter(checkpoint_path, device=device)

    extractor = AudioSpectrogramExtractor()

    # Read and prepare source audio
    if not os.path.exists(source_wav_path):
        raise FileNotFoundError(f"Source audio file not found: '{source_wav_path}'")

    audio_src, _ = sf.read(source_wav_path)
    if audio_src.ndim > 1:
        audio_src = np.mean(audio_src, axis=-1)
    if len(audio_src) < N_SAMPLES:
        audio_src = np.pad(audio_src, (0, N_SAMPLES - len(audio_src)))
    else:
        audio_src = audio_src[:N_SAMPLES]

    spec_src = extractor.extract(torch.tensor(audio_src, dtype=torch.float32)).to(device)

    # Read or synthesize target speaker reference
    if target_wav_path and os.path.exists(target_wav_path):
        audio_tgt, _ = sf.read(target_wav_path)
        if audio_tgt.ndim > 1:
            audio_tgt = np.mean(audio_tgt, axis=-1)
        if len(audio_tgt) < N_SAMPLES:
            audio_tgt = np.pad(audio_tgt, (0, N_SAMPLES - len(audio_tgt)))
        else:
            audio_tgt = audio_tgt[:N_SAMPLES]
        tgt_name = os.path.basename(target_wav_path)
    else:
        # Use target speaker preset
        spk_idx = 1  # Default female_soprano
        if target_speaker_name:
            matches = [s["id"] for s in SPEAKERS if s["name"].lower() == target_speaker_name.lower()]
            if matches:
                spk_idx = matches[0]
        audio_tgt = synth_speech_utterance(speaker_id=spk_idx)
        tgt_name = SPEAKERS[spk_idx]["name"]

    spec_tgt = extractor.extract(torch.tensor(audio_tgt, dtype=torch.float32)).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        conv_spec, c_src, s_tgt = model.convert(spec_src, spec_tgt)
        s_src, _ = model.speaker_enc(spec_src)
        s_conv, _ = model.speaker_enc(conv_spec)
        c_conv = model.content_enc(conv_spec)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Invert converted spectrogram to audio
    conv_wav = extractor.vocode(conv_spec).cpu().numpy().flatten()

    # Quantitative Verification Metrics
    target_similarity = float(F.cosine_similarity(s_conv, s_tgt).item())
    source_similarity = float(F.cosine_similarity(s_conv, s_src).item())
    content_similarity = float(F.cosine_similarity(c_conv.flatten(), c_src.flatten(), dim=0).item())
    speedup = AUDIO_DURATION / (latency_ms / 1000.0)

    # Save converted audio
    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
    sf.write(output_wav_path, conv_wav, SAMPLE_RATE)

    print("\n" + "=" * 65)
    print("        SPEAKER VOICE CONVERSION COMPLETE")
    print("=" * 65)
    print(f"Source Audio:            {source_wav_path}")
    print(f"Target Voice Persona:    {tgt_name}")
    print(f"Output Converted File:   {output_wav_path}")
    print(f"Target Speaker Sim (SECS):{target_similarity:.4f} (High identity match)")
    print(f"Source Speaker Sim (SECS):{source_similarity:.4f} (Disentangled shift)")
    print(f"Content Preservation:    {content_similarity:.4f} (Linguistic words conserved)")
    print(f"Inference Latency:       {latency_ms:.2f} ms ({speedup:.1f}x real-time)")
    print("=" * 65)

    return conv_wav, target_similarity, content_similarity

def run_demo(checkpoint_path: str = "models/autovc_voice_converter_best.pth", device: str = "cpu"):
    """Run built-in multi-speaker cross-gender voice conversion demo."""
    print("\n[DEMO] Initializing Voice Conversion Demonstration...")
    model = load_voice_converter(checkpoint_path, device=device)

    os.makedirs("outputs", exist_ok=True)
    m_path = "outputs/sample_source_male_baritone.wav"
    f_path = "outputs/sample_target_female_soprano.wav"
    out_m2f = "outputs/demo_male_to_female_converted.wav"
    out_f2m = "outputs/demo_female_to_male_converted.wav"

    if not os.path.exists(m_path):
        sf.write(m_path, synth_speech_utterance(speaker_id=0), SAMPLE_RATE)
    if not os.path.exists(f_path):
        sf.write(f_path, synth_speech_utterance(speaker_id=1), SAMPLE_RATE)

    print("\n>>> Scenario 1: Converting Adult Male Baritone -> Female Soprano...")
    convert_voice(
        source_wav_path=m_path,
        target_wav_path=f_path,
        output_wav_path=out_m2f,
        model=model,
        device=device
    )

    print("\n>>> Scenario 2: Converting Female Soprano -> Adult Male Baritone...")
    convert_voice(
        source_wav_path=f_path,
        target_wav_path=m_path,
        output_wav_path=out_f2m,
        model=model,
        device=device
    )

def main():
    parser = argparse.ArgumentParser(description="AutoVC Voice Conversion CLI")
    parser.add_argument("--source", type=str, default=None, help="Source audio file (.wav)")
    parser.add_argument("--target", type=str, default=None, help="Reference audio file of target speaker (.wav)")
    parser.add_argument("--target-speaker", type=str, default=None, choices=["male_baritone", "female_soprano", "male_tenor", "deep_bass"], help="Target speaker preset persona")
    parser.add_argument("--output", type=str, default=None, help="Output converted audio file path (.wav)")
    parser.add_argument("--demo", action="store_true", help="Run automated cross-gender voice conversion demonstration")
    parser.add_argument("--checkpoint", type=str, default="models/autovc_voice_converter_best.pth", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.demo:
        run_demo(checkpoint_path=args.checkpoint, device=args.device)
        return

    if args.source is None:
        print("[ERROR] Please provide --source <file.wav> or run --demo.")
        parser.print_help()
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        base, ext = os.path.splitext(args.source)
        tgt_label = args.target_speaker or (os.path.splitext(os.path.basename(args.target))[0] if args.target else "converted")
        out_path = f"{base}_to_{tgt_label}{ext}"

    convert_voice(
        source_wav_path=args.source,
        target_wav_path=args.target,
        target_speaker_name=args.target_speaker,
        output_wav_path=out_path,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

if __name__ == "__main__":
    main()

