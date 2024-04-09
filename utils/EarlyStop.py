import os
import time

import torch
import numpy as np
from .utils import char_color


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, dir='output', name='checkpoint', best_score=None, patience=7, delta=0,
                 mode='max', device='cpu'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode ('max'|'min'): 'max' means the larger the evaluation indicator, the better
        """
        self.patience = patience
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.name = name
        self.dir = dir
        self.mode = mode
        self.device = device
        try:
            os.makedirs(self.dir, exist_ok=True)
            os.chmod(self.dir, mode=0o777)
            print(f'{dir} created')
        except Exception:
            print(f"{dir} :EarlyStop make dir Error")

    def __call__(self, val_loss, model, optimizer, lr_scheduler):
        score = val_loss
        if self.best_score == torch.nan:
            print(f'{self.name}: there is nan in score')
            raise ValueError

        if self.best_score is None:
            self.best_score = score
            print("Early stop initiated")
            self.save_checkpoint(model, optimizer, lr_scheduler)
            return

        if self.mode == 'max':
            if score < self.best_score + self.delta:
                self.counter += 1
                print(
                    char_color(
                        f'{self.name} EarlyStopping counter: {self.counter} out of {self.patience}       best score: {self.best_score}'
                    )
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(char_color(f"{self.name} {self.best_score} -> {score}"))
                self.best_score = score
                self.save_checkpoint(model, optimizer, lr_scheduler)
                self.counter = 0
        elif self.mode == 'min':
            if score > self.best_score + self.delta:
                self.counter += 1
                print(
                    char_color(
                        f'{self.name} EarlyStopping counter: {self.counter} out of {self.patience}       best score: {self.best_score}'
                    )
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(char_color(f"{self.name} {self.best_score} -> {score}"))
                self.best_score = score
                self.save_checkpoint(model, optimizer, lr_scheduler)
                self.counter = 0

    def save_checkpoint(self, model, optimizer, lr_scheduler, best_score=None, name=None):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if name is None:
            name = self.name
        if best_score is None:
            best_score = self.best_score
        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_loss": best_score,
                "best_score": best_score,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                'lr': optimizer.param_groups[0]['lr'] if optimizer is not None else None,
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None
            },
            f"{os.path.join(self.dir, name)}"
        )

    def load_checkpoint(self, model, optimizer, lr_scheduler, ignore=False, cp=True, name=None):
        """
        :param model:
        :param ignore:
        :param cp: save a copy while load a checkpoint
        :return:
        """
        if name is None:
            name = self.name
        if os.path.exists(f'{self.dir}/{name}'):
            check_path = f'{self.dir}/{name}'
            checkpoint = torch.load(check_path, map_location=self.device)
            if cp:
                os.system(
                    f'cp {self.dir}/{name} {self.dir}/{name}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}'
                )
        else:
            check_path = f'checkpoint/{name}'
            if os.path.exists(check_path):
                checkpoint = torch.load(check_path, map_location=self.device)
            else:
                print(char_color(f"{self.dir}/{name} NOT EXISTS", color='r'))
                return False
        print_strs = f'[{name} load from {check_path}]'

        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            if not ignore:
                if "optimizer" in checkpoint.keys() and checkpoint["optimizer"] is not None:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        print_strs += ' optimizer loaded'
                    except Exception:
                        print(char_color('Load optimizer Failed', color='r'))
                if "lr_scheduler" in checkpoint.keys() and checkpoint["lr_scheduler"] is not None:
                    try:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                        print_strs += ' lr_scheduler loaded'
                    except Exception:
                        print(char_color('Load lr_scheduler Failed', color='r'))

                if "lr" in checkpoint.keys() and checkpoint['lr'] is not None:
                    print_strs += f' lr {checkpoint["lr"]}'

                self.best_score = checkpoint["best_score"]
        else:
            model.load_state_dict(checkpoint, strict=False)
        print_strs += f'score: {self.best_score}'
        print(char_color(print_strs))
        model.eval()
        return True


class MultiEarlyStop:
    """
    Use for save the best for every component of the model
    model should support magic method __getitem__
    """

    def __init__(self, train_tags: dict = None, best_tag=None, dir='result', best_score=None, patience=7,
                 delta=0, mode: dict = None, device='cpu'):
        """
        @param train_tags: used for the component want to save, should be some of the model's train_tags
        @param best_tag: used for the whole model to save, which part should be the best
        @param dir: target path to save
        @param best_score:
        @param patience:
        @param verbose: If True, prints a message for each validation loss improvement.
        @param delta:
        @param mode: mix or max
        @param device:
        """
        super(MultiEarlyStop, self).__init__()
        self.train_tags = train_tags
        mode = {k: 'max' for k in train_tags.keys()} if mode is None else mode
        self.stoppers = {t: EarlyStopping(dir=dir,
                                          name=f'{t}.pth',
                                          best_score=best_score,
                                          patience=patience,
                                          delta=delta,
                                          mode=mode[t],
                                          device=device
                                          ) for t in train_tags.keys()}
        self.time_stopper = EarlyStopping(dir=os.path.join(dir, 'times_stopper'),
                                          name=f'whole.pth',
                                          best_score=best_score,
                                          patience=patience,
                                          delta=delta,
                                          mode=mode[best_tag] if best_tag is not None else 'max',
                                          device=device
                                          )
        self.best_tag = best_tag

    def step(self, tag=None, val_record=None, lr_scheduler=None, model=None, optimizer: dict = None, epoch=0):
        if epoch % 2 == 0:
            self.time_stopper.save_checkpoint(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                              best_score=val_record[self.best_tag],
                                              name=f'whole_' + f'{epoch}'.zfill(3) + '.pth')
        if isinstance(tag, (list, tuple)):
            for t in tag:
                self.stoppers[t](val_loss=val_record[t], model=model[t], optimizer=optimizer[t],
                                 lr_scheduler=lr_scheduler[t])
                if t == self.best_tag:
                    self.time_stopper(val_loss=val_record[t], model=model, optimizer=optimizer,
                                      lr_scheduler=lr_scheduler)
        elif isinstance(tag, str) and tag in self.train_tags.keys():
            self.stoppers[tag](val_loss=val_record[tag], model=model[tag], optimizer=optimizer[tag],
                               lr_scheduler=lr_scheduler[tag])
            if tag == self.best_tag:
                self.time_stopper(val_loss=val_record[tag], model=model, optimizer=optimizer,
                                  lr_scheduler=lr_scheduler)
        elif tag is None:
            for tag in self.train_tags.keys():
                if self.train_tags[tag]:
                    self.stoppers[tag](val_loss=val_record[tag], model=model[tag], optimizer=optimizer[tag],
                                       lr_scheduler=lr_scheduler[tag])
                    if tag == self.best_tag:
                        self.time_stopper(val_loss=val_record[tag], model=model, optimizer=optimizer,
                                          lr_scheduler=lr_scheduler)
        else:
            raise ValueError

    def load_checkpoint(self, tag=None, model=None, optimizer=None, lr_scheduler=None, ignore=False, cp=True,
                        whole=False, idx=None):
        """
        :param tag: load checkpoint name, (if tag is 'gen', the file name is gen.pth)
        :param model:
        :param ignore:
        :param cp: copy the checkpoint in same dir
        :param whole: Load whole model, not part of it
        :param idx: only used when whole is True, load the checkpoint id == idx, if not exist then try to load the best
        :return:
        """
        load_flag = False
        if isinstance(tag, (list, tuple)):
            for t in tag:
                self.stoppers[t].load_checkpoint(model=model[t], optimizer=optimizer[t], lr_scheduler=lr_scheduler[t],
                                                 ignore=ignore, cp=cp)

        elif isinstance(tag, str) and tag in self.train_tags.keys():
            self.stoppers[tag].load_checkpoint(model=model[tag],
                                               optimizer=optimizer[tag],
                                               lr_scheduler=lr_scheduler[tag],
                                               ignore=ignore,
                                               cp=cp)
        elif not whole and idx is None:
            for tag in self.train_tags.keys():
                if self.stoppers[tag].load_checkpoint(model=model[tag],
                                                      optimizer=optimizer[tag],
                                                      lr_scheduler=lr_scheduler[tag],
                                                      ignore=ignore,
                                                      cp=cp):
                    load_flag = True
        # if no part checkpoint file, try to load whole
        if not load_flag:
            try:
                if idx is None and os.path.exists(os.path.join(self.time_stopper.dir, self.time_stopper.name)):
                    # load the best checkpoint
                    self.time_stopper.load_checkpoint(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                      ignore=ignore)
                else:
                    name = None
                    checkpoint_list = os.listdir(f'{self.time_stopper.dir}')
                    checkpoint_list.sort()
                    if len(checkpoint_list) == 0 and whole:
                        # if want to load whole but failed, try to load part
                        for tag in self.train_tags.keys():
                            if self.stoppers[tag].load_checkpoint(model=model[tag],
                                                                  optimizer=optimizer[tag],
                                                                  lr_scheduler=lr_scheduler[tag],
                                                                  ignore=ignore,
                                                                  cp=cp):
                                print('\033[1;31m', f'No Whole Checkpoint, Load {tag}', '\033[0m')
                        return
                    if idx is not None:
                        if idx == -1:
                            name = checkpoint_list[-1]
                        else:
                            for n in checkpoint_list:
                                if str(idx) in n:
                                    name = n
                                    break
                    if name is None:
                        name = sorted(os.listdir(f'{self.time_stopper.dir}'))[-1]
                    print(char_color(f'No Best Whole, Load {name}', color='g'))
                    self.time_stopper.load_checkpoint(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                      ignore=ignore, name=name, cp=False)
            except IndexError:
                print('time stopper not exists')

    @property
    def early_stop(self):
        return {k: e.early_stop or not self.train_tags[k] for k, e in self.stoppers.items()}

    @property
    def best_score(self):
        return {k: e.best_score for k, e in self.stoppers.items()}
