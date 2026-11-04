"""Training CLI; reusable implementation is in engine.trainer."""
import argparse

from ..runtime import compute_action_stats, load_config, make_initial_noise
from ..evaluation.metrics import compute_batch_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train SPOT from a config plus dotted overrides')
    parser.add_argument('--config')
    parser.add_argument('--resume')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.resume:
        cfg.train.resume = args.resume
    from ..engine.trainer import train
    return train(cfg)


if __name__ == '__main__':
    main()
