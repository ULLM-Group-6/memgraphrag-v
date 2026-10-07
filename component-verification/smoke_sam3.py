"""Exercise MG2's real SAM3 wrapper; run in its Linux/CUDA environment."""
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
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--prompts', nargs='+', default=['impala', 'airplane'])
    parser.add_argument('--output', type=Path, default=Path('outputs') / ('sam3_smoke_' + time.strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    image_path = (args.image or repo / 'examples/data/impala_demo/kb_image.jpg').resolve()
    for path in [checkpoint, image_path, repo / 'assets/bpe_simple_vocab_16e6.txt.gz',
                 repo / 'src/mmgraphrag/grounding_model/sam3Model.py']:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(args.prompts)) != len(args.prompts):
        raise ValueError('Use distinct prompts for this smoke test.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('MG2 SAM3Grounding requires CUDA; use an allocated GPU.')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    # Import the original wrapper under an isolated package name. This skips
    # mmgraphrag/__init__.py, which imports unrelated graph/CuPy components.
    # The SAM3 implementation, weights, inference and crop code are unchanged.
    sys.path.insert(0, str(repo))
    package = types.ModuleType('mg2_sam3_smoke')
    package.__path__ = [str(repo / 'src/mmgraphrag')]
    sys.modules[package.__name__] = package
    config_module = importlib.import_module('mg2_sam3_smoke.utils.config_utils')
    wrapper = importlib.import_module('mg2_sam3_smoke.grounding_model.sam3Model')
    import sam3
    if Path(sam3.__file__).resolve().parent != repo / 'sam3':
        raise RuntimeError('Imported SAM3 is not the copy bundled with this MG2 checkout.')

    config = config_module.BaseConfig(
        grounding_model_name=str(checkpoint),
        grounding_model_work_dir='grounding',
        grounding_model_prompt_batch_size=1,
        grounding_detection_threshold=0.5,
        sam3_device=0,
        save_dir=str(output / 'config'),
    )
    start = time.perf_counter()
    model = wrapper.SAM3Grounding(global_config=config, global_work_dir=str(output))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    width, height = image.size
    records = []

    # Repeat the real prompts, then exercise the deterministic empty-input path.
    for name, prompts in [('first', args.prompts), ('repeat', args.prompts), ('empty', [])]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(image=image, prompt=prompts, prompt_batch_size=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        masks = result['masks'].detach().cpu().numpy()
        boxes = result['boxes'].detach().cpu().numpy()
        scores = result['scores'].detach().cpu().numpy()
        labels = result['labels'].detach().cpu().numpy()
        count = len(scores)
        assert masks.shape == (count, 1, height, width), masks.shape
        assert boxes.shape == (count, 4) and labels.shape == (count,)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert ((scores >= 0.5 - 1e-6) & (scores <= 1 + 1e-6)).all()
        assert ((labels >= 0) & (labels < len(prompts))).all()
        if not prompts:
            assert count == 0

        crops = model.get_save_crop(result, chunk_id='smoke_passage',
                                    img_id=name, max_workers=1)
        save_dir = output / 'grounding' / 'smoke_passage' / name
        metadata = json.loads((save_dir / 'metadata.json').read_text())
        assert metadata['total_objects'] == len(crops)
        # MG2 deliberately discards boxes narrower/shorter than 10 pixels.
        eligible = 0
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            eligible += int(min(width, x2) - max(0, x1) >= 10 and
                            min(height, y2) - max(0, y1) >= 10)
        assert len(crops) == eligible, 'Expected crops were not saved; inspect MG2 error logs.'
        for crop in crops:
            with Image.open(crop['save_path']) as saved:
                saved.verify()
        np.savez_compressed(save_dir / 'raw_detections.npz', masks=masks,
                            boxes=boxes, scores=scores, labels=labels)
        preview = np.array(image).copy()
        for mask in masks[:, 0]:
            visible = mask.astype(bool)
            preview[visible] = (0.6 * preview[visible] + 0.4 * np.array([255, 60, 60])).astype('uint8')
        preview = Image.fromarray(preview)
        draw = ImageDraw.Draw(preview)
        for box, score, label in zip(boxes, scores, labels):
            draw.rectangle(box.tolist(), outline='red', width=3)
            draw.text((max(0, float(box[0])), max(0, float(box[1]))),
                      f'{prompts[int(label)]}: {float(score):.3f}', fill='red')
        preview.save(save_dir / 'overlay.png')
        # Fixture IDs verify associations without requiring MemGraph indexing.
        mappings = []
        for item in metadata['objects']:
            detection_index = item['local_id']
            mappings.append({
                'crop_path': str(save_dir / item['filename']),
                'entity_id': f'smoke_entity_{int(labels[detection_index]):03d}',
                'prompt': prompts[int(labels[detection_index])],
                'score': float(scores[detection_index]),
                'image_id': 'smoke_image',
            })
        record = dict(run=name, prompts=prompts, detections=count,
                      saved_crops=len(crops), inference_seconds=elapsed,
                      detections_per_prompt={p: int((labels == i).sum()) for i, p in enumerate(prompts)},
                      crop_entity_mappings=mappings)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'crop_entity_mappings'}, indent=2))

    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    summary = dict(mg2_commit=commit, checkpoint=str(checkpoint), sam3_module=sam3.__file__,
                   image=str(image_path), torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), load_seconds=load_seconds,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
                   threshold=0.5, prompt_batch_size=1, runs=records,
                   status='structural checks passed; visual inspection still required')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Output: {output}')
    print(summary['status'])
    if not records[0]['saved_crops']:
        print('No positive crop produced: positive grounding remains unverified.')


if __name__ == '__main__':
    main()
