# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH).parents[1]

analysis = Analysis(
    [str(project_root / "desktop_app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "app_version.txt"), "."),
        (str(project_root / "artifacts/final_heads.npy"), "artifacts"),
        (str(project_root / "artifacts/final_heads.json"), "artifacts"),
        (str(project_root / "artifacts/score_correction.json"), "artifacts"),
        (str(project_root / "experiments/final_holdout_plan.json"), "experiments"),
        (str(project_root / "data/previews/track_0207501_30s.wav"), "data/previews"),
    ],
    hiddenimports=[
        "predict_app",
        "score_correction",
        "safetensors.torch",
        "soundfile",
        "soxr",
        "transformers.audio_utils",
        "transformers.feature_extraction_sequence_utils",
        "transformers.feature_extraction_utils",
        "transformers.models.audio_spectrogram_transformer",
        "transformers.models.audio_spectrogram_transformer.configuration_audio_spectrogram_transformer",
        "transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["flax", "jax", "jaxlib", "tensorflow"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MusicClassification",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MusicClassification",
)
