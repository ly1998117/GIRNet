# third-party imports
import json
import math
import os

import torch.optim.lr_scheduler
import numpy as np

# project imports
from .utils import char_color, make_dirs, TXTLogs, CSVLogs, save_img
from .trainer import TrainEpoch, ValidEpoch, TestEpoch
from .dataloader import get_loader
from .EarlyStop import MultiEarlyStop
from params import load_data_keys, all_params
import warnings
from visualize.utils import Plot, plot_curve

warnings.filterwarnings("ignore")


class MultiDict:
    def __init__(self, item: dict):
        self.item = item

    def __getitem__(self, item):
        return self.item[item]

    def __setitem__(self, key, value):
        self.item[key] = value

    def keys(self):
        return self.item.keys()

    def values(self):
        return self.item.values()

    def items(self):
        return self.item.items()

    def state_dict(self):
        return {key: value.state_dict() for key, value in self.item.items()}

    def load_state_dict(self, state_dict):
        for key, value in self.item.items():
            value.load_state_dict(state_dict[key])

    @property
    def param_groups(self):
        return list(self.item.values())[0].param_groups


class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs):
        super().__init__(optimizer, lambda epoch: (1 - (epoch / epochs) ** 0.9))
        self.epochs = epochs

    def get_count(self):
        return self.last_epoch


class PloyLRs(MultiDict):
    """
    Multi-optimizer learning rate adjustment
    """

    def __init__(self, optimizer: dict, epochs):
        lrs = {}
        if isinstance(epochs, int):
            epochs = dict(map(lambda k: (k, epochs), optimizer.keys()))
        if isinstance(epochs, (list, tuple)):
            epochs = dict(zip(optimizer.keys(), epochs))
        [lrs.update({n: PolyLR(opt, epochs[n])}) for n, opt in optimizer.items()]
        super().__init__(lrs)

    def step(self, epoch: TrainEpoch):
        marks = epoch.train_state
        for n, m in marks.items():
            if n not in self.item.keys() or self.item[n].get_count() >= self.item[n].epochs:
                epoch.set_signal(n, False)
            if m:
                self.item[n].step()

    def get_epoch(self):
        return max([i.get_count() for i in self.item.values()])


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
            self, optimizer, warmup_steps: int, t_total: int, cycles: float = 0.5, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

    def get_count(self):
        return self.last_epoch


class WarmupCosineScheduled:
    """
    Multi-optimizer learning rate adjustment
    """

    def __init__(self, optimizer: dict, warmup_steps, epochs):
        self.lrs = {}
        if isinstance(epochs, int):
            epochs = dict(map(lambda k: (k, epochs), optimizer.keys()))
        if isinstance(epochs, (list, tuple)):
            epochs = dict(zip(optimizer.keys(), epochs))
        [self.lrs.update({n: WarmupCosineSchedule(opt, warmup_steps, epochs[n])}) for n, opt in optimizer.items()]

    def step(self, epoch: TrainEpoch):
        marks = epoch.train_state
        for n, m in marks.items():
            if n not in self.lrs.keys() or self.lrs[n].get_count() >= self.lrs[n].t_total:
                epoch.set_signal(n, False)
            if m:
                self.lrs[n].step()


def train(args, model, optimizer, lr_scheduler, loaders, stoppers, logger, metrics_logger):
    train_loader, val_loader = loaders

    if not hasattr(args, 'ncc_fn_none'):
        args.ncc_fn_none = None
    trainepoch = TrainEpoch(model, loss=args.losses, optimizer=optimizer, metrics=args.metrics, device=args.device,
                            verbose=True, load_data_keys=args.load_data_keys, pretrain=args.pretrain,
                            k=args.k, loop_flag=args.loop_flag, train_by_epoch=args.train_by_epoch,
                            plot_fn=Plot(model=model.model_id, ncc_fn=args.ncc_fn_none,
                                         dir_path=f'{args.output_dir}/{args.dir_name}/train_plot',
                                         epoch=lr_scheduler.get_epoch(),
                                         slices_n=args.slices_n,
                                         dpi=args.dpi,
                                         im_format=args.format))
    [trainepoch.set_signal(k, v) for k, v in args.train_tags.items()]
    validepoch = ValidEpoch(model, loss=args.losses, metrics=args.metrics, device=args.device, verbose=True,
                            load_data_keys=args.load_data_keys, pretrain=args.pretrain,
                            plot_fn=Plot(model=model.model_id, ncc_fn=args.ncc_fn_none,
                                         dir_path=f'{args.output_dir}/{args.dir_name}/val_plot',
                                         epoch=lr_scheduler.get_epoch(),
                                         slices_n=args.slices_n,
                                         dpi=args.dpi,
                                         im_format=args.format
                                         ))

    for i in range(lr_scheduler.get_epoch(), max(args.epochs.values())):
        # record learning rate
        for k in optimizer.keys():
            if k in args.pretrain.keys() or trainepoch.get_signal()[k] and all(
                    i >= np.array(list(args.pretrain.values()))):
                print(char_color(f"{k} [Epoch: {i}/{args.epochs[k]}], "
                                 f"lr: {optimizer[k].param_groups[0]['lr']}, path: {args.dir_name}")),
                logger.log(f"{k} [Epoch: {i}/{args.epochs[k]}], "
                           f"lr: {optimizer[k].param_groups[0]['lr']}, path: {args.dir_name}")
        # start epoch
        trainlogs = trainepoch(train_loader)
        validlogs = validepoch(val_loader)
        metrics_logger.cache({'stage': 'train', 'epoch': i, **trainlogs})
        metrics_logger.cache({'stage': 'valid', 'epoch': i, **validlogs})
        metrics_logger.write()

        logger.log(f'train: {trainlogs}')
        logger.log(f'val: {validlogs}')
        lr_scheduler.step(trainepoch)

        # early stopper update
        stoppers.step(
            tag=[args.metrics_tags_map[log] for log in validlogs.keys() if log in args.metrics_tags_map.keys()],
            val_record={args.metrics_tags_map[log]: v for log, v in validlogs.items() if
                        log in args.metrics_tags_map.keys()},
            lr_scheduler=lr_scheduler,
            model=model,
            optimizer=optimizer,
            epoch=i,
        )

        # set stopper result to epoch, example: stoppers.early_stop == {'reg' : True}, stop register training
        [trainepoch.set_signal(k, not v) for k, v in stoppers.early_stop.items()]
        if not all(stoppers.early_stop):
            break

    print(char_color(f"best dice {stoppers.best_score}"))
    logger.log(f"best dice {stoppers.best_score}")


def train_fold(args, i, model, opti):
    dir_name = args.dir_name
    loss_name = ''
    load_data_key = args.load_data_keys
    args.load_data_keys = load_data_keys[args.load_data_keys]
    # if isinstance(args.losses, (list, tuple)):
    #     for ls in args.losses:
    #         loss_name += f'_{ls.__name__}'
    args.dir_name += f'{loss_name}_{args.name}/{args.mark}/fold_{i}_{args.contrast}'
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    make_dirs(f'{args.output_dir}/{args.dir_name}')
    print(char_color(f'Using device {args.device}'))
    print(char_color(f'Using path {args.output_dir}/{args.dir_name}'))
    stoppers = MultiEarlyStop(train_tags=args.train_tags,
                              best_tag=args.best_tag,
                              dir=f'{args.output_dir}/{args.dir_name}',
                              patience=args.patience,
                              mode=args.modes,
                              device=args.device)
    lr_scheduler = PloyLRs(opti, args.epochs)
    model.to(device=args.device)
    metrics_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name='metrics')

    if args.resume:
        stoppers.load_checkpoint(model=model, optimizer=opti, lr_scheduler=lr_scheduler,
                                 ignore=args.ignore, idx=args.load_idx)
        metrics_logger.resume(key='epoch', value=lr_scheduler.get_epoch())

    if not args.test and max(args.epochs.values()) > 0:
        # Log
        logger = TXTLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name='param')

        for key, value in args.__dict__.items():
            logger.log(f'{key}: {value}')
        for key, value in all_params(args.task).items():
            logger.log(f'{key}: {value}')

        logger = TXTLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name='print')

        # training
        loader = get_loader(task=args.task, load_data_keys=args.load_data_keys, data_dir=args.data_dir, k_fold=i,
                            batch_size=args.bz, num_worker=args.num_works)
        train(args, model, opti, lr_scheduler, loader, stoppers, logger, metrics_logger)
        if metrics_logger.exist:
            plot_curve(f'{args.output_dir}/{args.dir_name}', metrics_logger)

    loader = get_loader(task=args.task, load_data_keys=args.load_data_keys, data_dir=args.data_dir, k_fold=i,
                        batch_size=args.bz, mode='val')
    if args.plot:
        plot_epoch(args, model, loader)
    if args.save_deformation:
        save_deformation_field_epoch(args, model, loader, idx=args.load_idx)
    if args.save_seg:
        save_seg_epoch(args, model, loader, idx=args.load_idx)
    del loader
    args.dir_name = dir_name
    args.load_data_keys = load_data_key
    del model


def test_epoch(args, model, loaders, plot_fn=(None, None), interface=(None, None), tags=None, name='Test'):
    """
    run one epoch with operation functions
    :param args: include (device, output_dir, dir_name, train_tags,
                modes, ignore, metrics, load_data_keys)
    :param model:
    :param loaders:
    :param plot_fn: (train_fn, valid_fn)
    :param interface: (train_fn, valid_fn)
    :return:
    """
    # from utils.trainer import TestEpoch
    # from utils.EarlyStop import MultiEarlyStop

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    make_dirs(f'{args.output_dir}/{args.dir_name}')
    print(char_color(f'[{name} Epoch]: Using device {args.device}'))
    print(char_color(f'[{name} Epoch]: Using path {args.output_dir}/{args.dir_name}'))
    train_loader, val_loader = loaders
    model.to(device=device)

    validepoch = TestEpoch(model, metrics=args.metrics, device=device, verbose=True,
                           load_data_keys=args.load_data_keys, interface=interface[1],
                           plot_fn=plot_fn[1],
                           tags=tags)
    validepoch(val_loader)
    del validepoch


def save_deformation_field_epoch(args, model, loaders, name='deformation_field', idx='98'):
    def interface(dir_path, model_id):
        def _interface(desc, f_n, inp, pred, norm):
            [pos, neg], x2y, y2x = pred['infer']
            dir_name = f_n[args.load_data_keys[0]]
            save_img(inp[0], os.path.join(dir_path, dir_name, 'x.nii.gz'))
            # save_img(inp[1], os.path.join(dir_path, dir_name, 'y.nii.gz'))
            save_img(x2y, os.path.join(dir_path, dir_name, 'x2y.nii.gz'))
            save_img(y2x, os.path.join(dir_path, dir_name, 'y2x.nii.gz'))
            save_img(pos, os.path.join(dir_path, dir_name, 'x2y_df.nii.gz'))
            save_img(neg, os.path.join(dir_path, dir_name, 'y2x_df.nii.gz'))
            with open(os.path.join(dir_path, dir_name, 'f_n.txt'), 'w') as f:
                json.dump(f_n, f)
            return pred

        return _interface

    interfaces = [
        interface(
            dir_path=f'{args.output_dir}/{args.dir_name}/{name}/{idx}/train',
            model_id=model.model_id,
        ),
        interface(
            dir_path=f'{args.output_dir}/{args.dir_name}/{name}/{idx}/valid',
            model_id=model.model_id,
        )
    ]
    if not hasattr(args, 'metrics'):
        args.metrics = None
    test_epoch(args, model=model, loaders=loaders, interface=interfaces, tags=('infer',), name=name)


def save_seg_epoch(args, model, loaders, name='segmentation', idx='98'):
    def interface(dir_path, model_id):
        def _interface(desc, f_n, inp, pred, norm):
            mask = (pred['gen'][0] > .5).float()
            dir_name = f_n[args.load_data_keys[0]]
            save_img(inp[0], os.path.join(dir_path, dir_name, 'x.nii.gz'))
            save_img(mask, os.path.join(dir_path, dir_name, 'mask.nii.gz'))
            with open(os.path.join(dir_path, dir_name, 'f_n.txt'), 'w') as f:
                json.dump(f_n, f)
            return pred

        return _interface

    interfaces = [
        interface(
            dir_path=f'{args.output_dir}/{args.dir_name}/{name}/{idx}/train',
            model_id=model.model_id,
        ),
        interface(
            dir_path=f'{args.output_dir}/{args.dir_name}/{name}/{idx}/valid',
            model_id=model.model_id,
        )
    ]
    if not hasattr(args, 'metrics'):
        args.metrics = None
    test_epoch(args, model=model, loaders=loaders, interface=interfaces, tags=('gen',), name=name)


def plot_epoch(args, model, loaders):
    from utils.post_process import RandomWalk
    from visualize.utils import Plot
    random_walker = RandomWalk(crop=True, expand=30, beta=1000, fp=0.4, bp=0.8, kernel_size=5)

    def plot_interface(desc, f_n, inp, pred, norm):
        if 'gen' in pred.keys() and pred['gen'] is not None:
            mask = pred['gen'][0]
            moving = inp[0]
            pred['gen'][0] = random_walker(moving, mask)
        return pred

    plot_fn = (
        Plot(
            model=model.model_id,
            dir_path=f'{args.output_dir}/{args.dir_name}/test_plot/train',
            save_img=True,
            slices_n=args.slices_n,
            dpi=args.dpi,
            im_format=args.format
        ),
        Plot(
            model=model.model_id,
            dir_path=f'{args.output_dir}/{args.dir_name}/test_plot/valid',
            save_img=True,
            slices_n=args.slices_n,
            dpi=args.dpi,
            im_format=args.format
        )
    )
    interfaces = (
        plot_interface, plot_interface
    )
    test_epoch(args, model=model, loaders=loaders, plot_fn=plot_fn, interface=interfaces)
