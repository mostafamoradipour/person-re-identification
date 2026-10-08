# Person Re-Identification

A research codebase for person re-identification (Re-ID): matching a person observed in one camera view with images or tracklets from other cameras. The repository includes image- and video-based training pipelines, model variants, evaluation utilities, and a demo.

## What’s included

- Image Re-ID training with PCB-style part-based classifiers and optional LSTM or graph-based components.
- Video Re-ID training with frame-sequence sampling.
- Dataset utilities and identity-balanced sampling.
- Evaluation scripts, including GPU and re-ranking variants.
- A demo for running a trained model.

## Requirements

The scripts use Python and PyTorch with torchvision, NumPy, and Matplotlib. A CUDA-capable GPU is recommended for training. This repository does not include a pinned dependency file; install compatible versions for your Python and CUDA environment.

## Data

Training scripts refer to Market-1501 and MARS data. Obtain each dataset through its official access process and prepare it in the directory layout expected by the dataset loader. The training code contains dataset roots and loader assumptions that may need adjustment for your local layout.

## Training

The image training entry point is `train_irid.py`; video training is `train_vrid.py`. Both expose command-line options for GPU selection, batch size, model name, backbone, and training configuration.

```bash
python train_irid.py --help
python train_vrid.py --help
```

The image pipeline initializes Market-1501 through the included dataset utilities. Review and update the dataset root in the script before launching training if your data is stored elsewhere. Example:

```bash
python train_irid.py --gpu_ids 0 --name experiment
```

Video training uses MARS and samples a configurable number of frames per sequence:

```bash
python train_vrid.py --gpu_ids 0 --seq_len 4 --sample_method random
```

Training writes checkpoints and logs according to the paths configured in the scripts.

## Evaluation and demo

Inspect the available options before running a script:

```bash
python evaluate.py --help
python evaluate_gpu.py --help
python evaluate_rerank.py --help
python demo.py --help
```

Evaluation and demo runs require compatible trained weights and correctly prepared query/gallery data. Checkpoint files are not guaranteed to be included in a clone.

## Repository layout

- `train_irid.py`, `train_vrid.py`: image and video training.
- `test_irid.py`, `test_vrid.py`: Re-ID evaluation pipelines.
- `evaluate*.py`: evaluation variants.
- `models/`: model definitions.
- `datasets/`: dataset loading and sampling utilities.
- `utils/`: losses and training helpers.
- `re_ranking.py`: re-ranking utility.

## Notes

This is an experimental research repository. Results depend on dataset preparation, checkpoint selection, and the software/hardware environment. No benchmark scores are claimed here.

## License

No license file is currently listed. Contact the repository owner before reusing or redistributing this code.
