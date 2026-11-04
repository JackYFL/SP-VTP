"""Run a manifest of checkpoint/condition evaluations without changing training."""
import argparse
import gc
import json
from pathlib import Path
from time import time

from .rollout import main as rollout_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(manifest['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    status = dict(stage='running', completed_jobs=[], completed_evaluations=0,
                  expected_evaluations=sum(len(job['interventions'])*3 for job in manifest['jobs']))
    started = time()
    try:
        for job in manifest['jobs']:
            status['current_job'] = f"seed_{job['seed']}/{job['name']}"
            status['elapsed_seconds'] = time()-started
            (root / 'suite_status.json').write_text(json.dumps(status, indent=2))
            arguments = ['--ckpt', job['checkpoint'], '--output-dir', str(root / status['current_job']),
                         '--interventions', ','.join(job['interventions'])]
            for key, value in manifest['common'].items():
                arguments.extend(['--'+key.replace('_','-'), str(value)])
            print(f"START {status['current_job']}", flush=True)
            results = rollout_main(arguments)
            for summaries in results.values():
                assert summaries['all']['training_seed'] == job['seed']
                assert summaries['all']['episodes'] == manifest['expected_episodes']
            status['completed_jobs'].append(status['current_job'])
            status['completed_evaluations'] += len(results)
            print(f"DONE {status['current_job']}: {status['completed_evaluations']}/{status['expected_evaluations']}", flush=True)
            gc.collect()
        assert status['completed_evaluations'] == status['expected_evaluations']
        status['stage'] = 'complete'
    except Exception as error:
        status.update(stage='failed', error=repr(error))
        raise
    finally:
        status['elapsed_seconds'] = time()-started
        (root / 'suite_status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
