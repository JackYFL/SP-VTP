"""Compatibility entry point for the shared iPhone processing pipeline."""
import sys
from .iphone_pipeline import *
from .iphone_pipeline import main as run

def main(cfg=None):
    if cfg is not None:
        return run(cfg)
    from .pipeline import main as cli
    return cli(["--backend", "iphone-oak", *sys.argv[1:]])

main.__wrapped__ = run

if __name__ == '__main__':
    main()
