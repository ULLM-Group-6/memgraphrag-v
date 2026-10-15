"""Smoke-test the real MG2 EvaClipModel adapter on one allocated GPU."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('external/MG2-RAG'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/eva-smoke/EVA-CLIP-8B'))
    parser.add_argument('--processor', type=Path, default=Path('checkpoints/eva-smoke/clip-processor'))
    parser.add_argument('--images', nargs='+', type=Path)
    parser.add_argument('--texts', nargs='+', default=[
        'a photograph of an impala antelope',
        'a photograph of an airplane',
        'a line chart showing revenue over time',
    ])
    parser.add_argument('--output', type=Path, default=Path('outputs') / ('eva_smoke_' + time.strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    repo, checkpoint, processor = args.repo.resolve(), args.checkpoint.resolve(), args.processor.resolve()
    expected_commit = '91f0eed6fc5f4383f0bae746d12850f05e381e77'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != expected_commit:
        raise RuntimeError(f'Expected MG2 commit {expected_commit}, got {commit}')
    for required in [checkpoint / 'config.json', checkpoint / 'pytorch_model.bin.index.json',
                     processor / 'preprocessor_config.json']:
        if not required.is_file():
            raise FileNotFoundError(required)
    index = json.loads((checkpoint / 'pytorch_model.bin.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    paths = args.images or [repo / 'examples/data/impala_demo/kb_image.jpg',
                           repo / 'examples/data/impala_demo/query_image.jpg']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible. Run inside an allocated GPU job.')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    images = []
    for path in paths:
        # Validate here: the upstream adapter silently substitutes black images
        # for unreadable paths, which must not count as a successful smoke test.
        with Image.open(path) as source:
            images.append(source.convert('RGB'))

    # Skip MG2's package initializer (which imports graph/SAM3/CuPy modules).
    # Import and exercise the original, unmodified EVA adapter and its helpers.
    package = types.ModuleType('mg2_eva_smoke')
    package.__path__ = [str(repo / 'src/mmgraphrag')]
    sys.modules[package.__name__] = package
    config_module = importlib.import_module('mg2_eva_smoke.utils.config_utils')
    adapter_module = importlib.import_module('mg2_eva_smoke.embedding_model.eva_clip')
    config = config_module.BaseConfig(
        multimodal_embedding_model_name=str(checkpoint),
        clip_image_processor_name=str(processor), clip_device=0,
        multimodal_embedding_batch_size=1, multimodal_embedding_max_model_len=77,
        save_dir=str(output),
    )
    start = time.perf_counter()
    encoder = adapter_module.EvaClipModel(global_config=config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    def timed_encode(**kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = encoder.batch_encode(**kwargs)
        torch.cuda.synchronize()
        return result, time.perf_counter() - start

    text_vectors, text_seconds = timed_encode(texts=args.texts)
    image_vectors, image_seconds = timed_encode(images=images)
    repeated, repeat_seconds = timed_encode(images=images[:1])
    for name, array, rows in [('text', text_vectors, len(args.texts)),
                              ('image', image_vectors, len(images)),
                              ('repeat', repeated, 1)]:
        assert array.shape == (rows, encoder.embedding_dim), (name, array.shape)
        assert np.isfinite(array).all(), f'{name}: nonfinite features'
        assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=5e-3), f'{name}: vectors not normalized'
    similarities = image_vectors @ text_vectors.T
    assert np.isfinite(similarities).all()
    assert (np.abs(similarities) <= 1.01).all()
    repeat_delta = float(np.max(np.abs(image_vectors[0] - repeated[0])))
    rankings = [np.argsort(-row, kind='stable').tolist() for row in similarities]
    np.savez(output / 'embeddings.npz', images=image_vectors, texts=text_vectors)
    revisions_path = checkpoint.parent / 'revisions.json'
    revisions = json.loads(revisions_path.read_text()) if revisions_path.is_file() else None
    summary = {
        'status': 'structural checks passed; inspect semantic rankings',
        'mg2_commit': commit, 'checkpoint_revisions': revisions,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'embedding_dimension': encoder.embedding_dim,
        'images': [str(p.resolve()) for p in paths], 'texts': args.texts,
        'cosine_similarities': similarities.tolist(), 'ranking_indices': rankings,
        'load_seconds': load_seconds, 'text_encode_seconds': text_seconds,
        'image_encode_seconds': image_seconds, 'repeat_image_seconds': repeat_seconds,
        'repeat_max_absolute_difference': repeat_delta,
        'repeat_close_at_1e-3': bool(np.allclose(image_vectors[:1], repeated, atol=1e-3, rtol=1e-3)),
        'peak_allocated_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
        'text_token_lengths': [len(encoder.tokenizer(t)['input_ids']) for t in args.texts],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
