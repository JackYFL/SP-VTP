"""Accelerate training; all final metrics reuse evaluation.runner."""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf

from ..evaluation.metrics import compute_batch_metrics, display_metrics, summarize_records
from ..evaluation.runner import evaluate_dataset
from ..runtime import (build_policy, compute_action_stats, make_dataset, make_initial_noise,
                       make_loader, save_checkpoint, select_episodes, write_split_manifest)


@torch.no_grad()
def validate(accelerator, policy, loader, dataset, max_batches=None):
    model = accelerator.unwrap_model(policy)
    was_training = policy.training
    policy.eval()
    groups = defaultdict(list)
    try:
        for number, batch in enumerate(loader):
            if max_batches is not None and number >= max_batches:
                break
            indices = batch.pop('sample_index')
            noise = make_initial_noise(indices.tolist(), dataset.action_horizon, 10)
            pred = model.predict_action(batch, initial_noise=noise)
            metrics = compute_batch_metrics(pred.float(), batch['future_actions'].float())
            gathered = accelerator.gather_for_metrics(dict(sample_index=indices, **metrics))
            gathered = {key: value.cpu().tolist() for key, value in gathered.items()}
            for bi, index in enumerate(gathered.pop('sample_index')):
                ep = dataset.episodes[dataset._samples[index][0]]
                row = dict(episode_key=ep.key, **{key: value[bi] for key, value in gathered.items()})
                groups[ep.scene].append(row)
                groups['all'].append(row)
        return {scene: summarize_records(rows) for scene, rows in groups.items()}
    finally:
        policy.train(was_training)


def train(cfg):
    cpu = str(cfg.train.get('device', 'auto')) == 'cpu'
    accelerator = Accelerator(cpu=cpu,
        mixed_precision='no' if cpu or not torch.cuda.is_available() else cfg.train.mixed_precision,
        gradient_accumulation_steps=int(cfg.train.grad_accum_steps),
        log_with='wandb' if cfg.train.wandb.enabled else None,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(int(cfg.train.seed))
    output = Path(cfg.train.output_dir)
    episodes = select_episodes(cfg, 'all')
    train_eps = select_episodes(cfg, 'train', cfg.data.get('split_manifest'), episodes=episodes)
    val_eps = select_episodes(cfg, 'val', cfg.data.get('split_manifest'), episodes=episodes)
    train_ds, val_ds = make_dataset(cfg, train_eps), make_dataset(cfg, val_eps)
    if not len(train_ds) or not len(val_ds):
        raise ValueError('Training and validation splits must both contain usable samples')
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / 'config.yaml')
        write_split_manifest(output / 'split_manifest.json', cfg, train_eps, val_eps)
    accelerator.wait_for_everyone()
    if cfg.train.wandb.enabled:
        accelerator.init_trackers(cfg.train.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={'wandb': dict(name=cfg.train.wandb.run_name, entity=cfg.train.wandb.entity)})
    resume = cfg.train.get('resume')
    policy = build_policy(cfg, from_checkpoint=bool(resume))
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False) if resume else {}
    if resume:
        policy.load_state_dict(checkpoint['policy'], strict=True)
    else:
        policy.set_normalizer(*compute_action_stats(train_ds, int(cfg.train.stats_samples), int(cfg.train.seed)))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    policy, optimizer, train_loader, val_loader = accelerator.prepare(policy, optimizer,
        make_loader(train_ds, int(cfg.train.batch_size), int(cfg.data.num_workers), shuffle=True),
        make_loader(val_ds, int(cfg.train.batch_size), int(cfg.data.num_workers), indexed=True))
    steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    total_steps = steps_per_epoch * int(cfg.train.num_epochs)
    sched = cfg.train.scheduler
    kwargs = dict(name=sched.name, optimizer=optimizer, num_training_steps=total_steps,
                  num_warmup_steps=int(sched.warmup_steps or total_steps * float(sched.warmup_ratio)))
    if sched.name in ('cosine', 'cosine_with_restarts'):
        kwargs['num_cycles'] = float(sched.num_cycles)
    scheduler = get_scheduler(**kwargs)
    if checkpoint.get('scheduler'):
        scheduler.load_state_dict(checkpoint['scheduler'])
    # Scheduler construction changes param-group LRs; restore the optimizer last
    # so the very first resumed update uses the checkpoint's actual LR/state.
    if checkpoint.get('optimizer'):
        optimizer.load_state_dict(checkpoint['optimizer'])
    step = int(checkpoint.get('step', 0))
    epoch = int(checkpoint.get('next_epoch', 0))
    max_steps = cfg.train.get('max_steps')
    accelerator.print(f'train={len(train_ds)} val={len(val_ds)} samples; '
        f'global batch={int(cfg.train.batch_size)*accelerator.num_processes*accelerator.gradient_accumulation_steps}')
    try:
        for epoch in range(epoch, int(cfg.train.num_epochs)):
            policy.train()
            if hasattr(train_loader, 'set_epoch'):
                train_loader.set_epoch(epoch)
            for batch in train_loader:
                with accelerator.accumulate(policy):
                    loss = policy(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    step += 1
                    if step % int(cfg.train.log_every) == 0:
                        logs = {'train/loss': float(loss.detach()), 'train/lr': scheduler.get_last_lr()[0]}
                        accelerator.log(logs, step=step)
                        accelerator.print(f'epoch={epoch+1} step={step} loss={float(loss.detach()):.5f}')
                    if max_steps and step >= int(max_steps):
                        break
            if (epoch + 1) % int(cfg.train.val_every_epochs) == 0:
                summaries = validate(accelerator, policy, val_loader, val_ds, cfg.train.val_max_batches)
                logs = {f'val/{scene}/{key}': value for scene, summary in summaries.items()
                        for key, value in display_metrics(summary).items()}
                accelerator.log(logs, step=step)
                accelerator.print(json.dumps(logs))
            if (epoch+1) % int(cfg.train.save_every_epochs) == 0 and accelerator.is_main_process:
                save_checkpoint(output / f'ckpt_ep{epoch:04d}.pt', accelerator.unwrap_model(policy), cfg,
                    next_epoch=epoch+1, epoch=epoch, step=step, optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict())
            if max_steps and step >= int(max_steps):
                break
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model = accelerator.unwrap_model(policy)
            save_checkpoint(output / 'ckpt_final.pt', model, cfg, next_epoch=epoch+1,
                step=step, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            if cfg.train.get('run_final_eval', True):
                summaries, _ = evaluate_dataset(model, val_ds, output / 'eval_metrics',
                    batch_size=int(cfg.train.batch_size), num_workers=int(cfg.data.num_workers),
                    metadata=dict(split='val', split_seed=int(cfg.data.split_seed),
                                  val_ratio=float(cfg.data.val_ratio)))
                accelerator.print(json.dumps(display_metrics(summaries['all'])))
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()
    return output / 'ckpt_final.pt'
