"""Compatibility entry point for the unified capture session."""
import sys
from spvtp.demonstration_collection.session import main

def record(argv=None):
    return main(["--backend", "iphone-oak", *(sys.argv[1:] if argv is None else argv)])

if __name__ == "__main__":
    record()
