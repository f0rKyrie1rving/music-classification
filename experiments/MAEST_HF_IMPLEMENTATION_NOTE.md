# MAEST official-source implementation change

The initial protocol selected the official UPF Essentia TensorFlow graph. Before
any MAEST weight, embedding, classifier, or development-validation outcome was
obtained, both the command-line downloader and the controlled browsers failed
to reach `essentia.upf.edu` from this machine. No completed metadata or model
file was produced and no MAEST plan JSON was frozen.

The MAEST authors' official GitHub repository states that their models are also
available on Hugging Face. The `mtg-upf` organization identifies itself as the
Music Technology Group at Universitat Pompeu Fabra and publishes the same
`discogs-maest-30s-pw-129e-519l` architecture as a 348 MB safetensors model.
This is therefore an alternate official distribution, not a third-party mirror.

The v2 implementation changes the artifact and runtime only:

- pin official Hugging Face revision
  `6c35f32a350f74351870937d5ae0bae1d898d1df`;
- use its SHA-256-addressed `model.safetensors` file rather than the inaccessible
  TensorFlow protobuf;
- use the pinned official MAEST feature-extractor configuration with local-only
  loading after review;
- decode exactly the first 30 source seconds with `soundfile`, average channels,
  and resample to 16 kHz with the pinned `soxr` HQ implementation before passing
  the waveform to that feature extractor;
- extract the same seventh-transformer-block 2,304-wide concatenation of CLS,
  DIST, and mean signal tokens specified in the original protocol.

The 903/303 data split, 329 excluded test IDs, broad-label ontology, folds,
classifier grid, OOF threshold policies, and decision rule remain unchanged.
This correction is frozen before any MAEST downstream outcome.
