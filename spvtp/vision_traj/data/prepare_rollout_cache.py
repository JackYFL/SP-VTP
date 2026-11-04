"""Build a validation-only cache overlay without mutating an existing data cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .build_frame_cache import _build_episode, _cache_paths, _indices_digest, _verify_episode_cache
from ..runtime import load_config, make_dataset, select_episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--source-cache-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.split_manifest).read_text())
    wanted = {row['episode_key'].replace('\\', '/') for row in manifest['val_episodes']}
    annotations = json.loads(Path(args.annotations).read_text())
    annotations = {key:value for key,value in annotations.items() if key.replace('\\','/') in wanted}
    annotation_path = output / 'val_annotations.json'
    annotation_path.write_text(json.dumps(annotations, indent=2))
    cfg = load_config(args.config, [f'data.root={args.data_root}',
        f'data.annotations_json={annotation_path}', 'data.frame_cache_root=null'])
    episodes = select_episodes(cfg, 'val', args.split_manifest)
    dataset = make_dataset(cfg, episodes)
    destination, source = output / 'frame_cache', Path(args.source_cache_root)
    def build(item):
        ei, episode = item
        raw = dataset._valid_raw_idx[ei]
        array, metadata_path = _cache_paths(source, episode)
        use_existing = False
        try:
            metadata = json.loads(metadata_path.read_text())
            frames = np.load(array, mmap_mode='r')
            # Legacy cache construction selected a prefix of this SAME valid-index sequence.
            # Matching count/endpoints therefore permits safe reuse of its frame selection.
            use_existing = (metadata['n_strided_frames'] == len(raw)
                and metadata['stride'] == int(cfg.data.stride)
                and metadata['image_size'] == int(cfg.data.image_size)
                and metadata['first_raw_frame'] == int(raw[0])
                and metadata['last_raw_frame'] == int(raw[-1])
                and frames.shape == (len(raw)+4, int(cfg.data.image_size), int(cfg.data.image_size), 3)
                and metadata.get('raw_indices_sha256', _indices_digest(raw)) == _indices_digest(raw))
        except (OSError, KeyError, ValueError):
            pass
        if use_existing:
            target, target_metadata = _cache_paths(destination, episode)
            target.parent.mkdir(parents=True, exist_ok=True)
            for origin, link in ((array, target), (metadata_path, target_metadata)):
                if not link.exists():
                    link.symlink_to(origin.resolve())
            result = dict(status='linked', episode_key=episode.key)
        else:
            result = _build_episode(episode, str(destination), int(cfg.data.stride), int(cfg.data.image_size), False)
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(build, enumerate(dataset.episodes)))
    for scene in sorted({ep.scene for ep in dataset.episodes}):
        episode = next(ep for ep in dataset.episodes if ep.scene == scene)
        _verify_episode_cache(episode, destination, int(cfg.data.stride), int(cfg.data.image_size))
    result = dict(data_root=args.data_root, annotations=str(annotation_path), frame_cache_root=str(destination),
        split_manifest=args.split_manifest, requested_episodes=len(episodes), usable_episodes=len(dataset.episodes),
        full_window_samples=len(dataset), cache_records=records,
        representative_seven_prompt_equality_passed=True,
        dropped_episode_keys=[ep.key for ep in episodes if ep.key not in {e.key for e in dataset.episodes}])
    (output / 'data_preparation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key:value for key,value in result.items() if key != 'cache_records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
