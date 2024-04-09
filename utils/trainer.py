import os
import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np


class AverageValueMeter(object):
    """
    Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        """
        Log a new value to the meter
        @param value: Next result to include.
        @param n: number
        """
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = 0, 0
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        """
        Get the value of the meter in the current state.
        """
        return self.mean, self.std

    def reset(self):
        """
        Resets the  meter to default settings.
        """
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = 0.0
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = 0.0


class Epoch:
    """
        Base Class for Training, use method __call__() or run() to start one epoch
    """

    def __init__(self, model, loss, metrics: list or tuple = None, stage_name: str = 'train',
                 device='cpu', verbose=True, load_data_keys=('moving_t1', 'atlas', 'moving_seg'),
                 plot_fn=None, lr_schedule=None, plot_by_epoch=False, tags=None):
        """
        @param model:
        @param loss: (tuple or list or instance)
        @param metrics: (instance that have __name__ member) show metrics in training
        @param stage_name: (str) 'train' | 'val' | 'test'
        @param device:
        @param verbose:
        @param load_data_keys: MRI load_data_keys
        """
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.load_data_keys = load_data_keys
        self.plot_fn = plot_fn
        self.plot_by_epoch = plot_by_epoch
        self.tags = self.model.tags if tags is None else tags
        self.f_name = {}
        if isinstance(self.model, list) or isinstance(self.model, tuple):
            [m.to(self.device) for m in self.model]
        elif isinstance(self.model, dict):
            [m.to(self.device) for m in self.model.values()]
        else:
            self.model.to(self.device)

        self.loss = self._to_dict(self.loss)
        self.lr_schedule = lr_schedule

    def _to_dict(self, item):
        item_dict = {}
        if hasattr(item, 'to'):
            item = [item]

        if isinstance(item, list) or isinstance(item, tuple):
            for i in item:
                k = [tag for tag in self.tags if tag in i.__name__]
                if len(k) == 0:
                    item_dict[i.__name__] = i
                else:
                    item_dict[k[0]] = i
            item = item_dict
        elif isinstance(item, dict):
            item = {n: l.to(self.device) for n, l in item.items()}
        else:
            return None
        return item

    def _to_cpu(self, x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.cpu()
        r = []
        for i in x:
            if isinstance(i, (list, tuple)):
                r.append(self._to_cpu(i))
            elif isinstance(i, torch.Tensor):
                r.append(i.to('cpu'))
            else:
                r.append(i)
        return r

    def _format_logs(self, logs):
        str_logs = [f'{k} : {v:.4}' for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, *param):
        """
        in this method, loss should be computed and gradient should be backward
        @param param: data
        @return:
        """
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def plot(self, *param):
        if self.plot_fn is not None:
            self.plot_fn(*param)

    def on_batch_start(self, data):
        """
        preprocess of input data
        @param data: data loader input
        @return: data file name, input data to model, normalization info (used to restore data)
        """
        f_name = data[f'{self.load_data_keys[0]}_meta_dict']['filename_or_obj'][0].split('/')[-2]
        self.f_name.update({key: data[f'{key}_meta_dict']['filename_or_obj'][0].split('/')[-2]
                            for key in self.load_data_keys})
        inp = []

        # used for reverse z-normalization
        norm = []
        for key in self.load_data_keys:
            norm_key = f'{key}_norm'
            if key in data.keys():
                inp.append(data[key].to(self.device))
            else:
                inp.append(None)
            if norm_key in data.keys():
                norm.append(data[norm_key].to(self.device))
            else:
                norm.append(None)
        return f_name, inp, norm

    def run(self, dataloader):
        return self(dataloader)

    def __call__(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meters = {loss.__name__: AverageValueMeter() for loss in
                       self.loss.values()} if self.loss is not None else None
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in
                          self.metrics} if self.metrics is not None else None

        with tqdm(dataloader, file=sys.stdout, disable=not self.verbose) as iterator:
            for idx, data in enumerate(iterator):
                f_name, inp, norm = self.on_batch_start(data)
                losses, pred = self.batch_update(f_name, inp, norm)
                if self.lr_schedule is not None:
                    self.lr_schedule.step(self)
                with torch.no_grad():
                    inp = self._to_cpu(inp)
                    if self.loss is not None:
                        # update loss logs
                        for tag, loss in self.loss.items():
                            if losses[tag] is not None:
                                iterator.set_description_str(desc=f'[{self.stage_name} {tag}]')
                                loss_value = losses[tag].cpu().detach().squeeze().numpy().item()
                                loss_meters[loss.__name__].add(loss_value)
                            if self.stage_name == 'test':
                                logs.update({loss.__name__: loss_value})
                            else:
                                logs.update({loss.__name__: loss_meters[loss.__name__].mean})
                    del losses

                    # update metrics logs
                    if self.metrics is not None:
                        for metric_fn in self.metrics:
                            if metric_fn.tag in pred.keys() and pred[metric_fn.tag] is not None:
                                metric_value = metric_fn(pred, inp).cpu().detach().squeeze().numpy().item()
                                metrics_meters[metric_fn.__name__].add(metric_value)
                                if self.stage_name == 'test':
                                    logs.update({metric_fn.__name__: metric_value})
                                else:
                                    logs.update({metric_fn.__name__: metrics_meters[metric_fn.__name__].mean})
                    if self.stage_name == 'test' or self.plot_by_epoch:
                        self.plot(logs, f_name, inp, pred, norm)
                    elif idx == len(iterator) - 1:
                        self.plot(logs, f_name, inp, pred, norm)

                    del pred, inp, norm
                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
        self.on_epoch_end()
        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer: dict, device='cpu', verbose=True,
                 load_data_keys=('moving_t1', 'atlas', 'moving_seg'),
                 pretrain: dict = None, k: int = 0, loop_flag=False, plot_fn=None,
                 train_by_epoch=False, tags=None):
        """
        @param pretrain: pretrain first net
        @param k: each net train k iteration
        @param loop_flag: if true, gen and inp would backward in one iteration
        @param train_by_epoch: gen or inp train complete epoch separately
        """
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            load_data_keys=load_data_keys,
            plot_fn=plot_fn,
            tags=tags
        )
        # control training stage
        self.signals = dict(zip(self.tags, [True] * len(self.tags)))
        self.optimizer = optimizer
        self.pretrain = pretrain if pretrain is not None else dict(zip(self.tags, [0] * len(self.tags)))
        # show training state
        self._train_state = None
        self.k = k if isinstance(k, dict) else {'gen': k, 'inp': k // 3}
        self._k = self.k.copy()
        self.loop_flag = loop_flag if loop_flag is False else None
        self.epoch_finish = False
        self.train_by_epoch = train_by_epoch

        def get_id(_k):
            k = _k
            while True:
                if k <= 0:
                    k = _k
                k -= 1
                yield k

        self.id_iter = get_id(len(self.tags))
        self.tag = self._get_current_tag()

    def _get_current_tag(self):
        """
        loop the tags find next trainable tag
        :return:
        """
        tag = self.tags[next(self.id_iter)]
        if not self.signals[tag] or tag not in self.k.keys():
            tag = self._get_current_tag()
        return tag

    def _fresh_global_current_tag(self):
        """
        :return:
        """
        self._k[self.tag] -= 1
        if self._k[self.tag] <= 0:
            self._k[self.tag] = self.k[self.tag]
            self.tag = self._get_current_tag()

    def get_signal(self):
        """
        get mark: which net has been trained in last iteration
        """
        return self.signals

    @property
    def train_state(self):
        return self._train_state

    def set_signal(self, tag, signal):
        self.signals[tag] = signal

    def on_epoch_end(self, *param):
        self.epoch_finish = True

    def on_epoch_start(self):
        self.model.train()
        self._train_state = dict(zip(self.tags, [False] * len(self.tags)))
        for k in self.pretrain.keys():
            if self.pretrain[k] >= 0:
                self.pretrain[k] -= 1

    def is_trainable(self, tag):
        """
        @param tag:
        @return: current stage trainable
        """

        if not self.signals[tag]:
            return False
        if tag in self.pretrain.keys() and self.pretrain[tag] >= 0:
            return True
        for k in self.pretrain.keys():
            if self.pretrain[k] > 0:
                return False

        if self.train_by_epoch:
            if self.epoch_finish:
                self._fresh_global_current_tag()
                self.epoch_finish = False
            if tag != self.tag:
                return False
            return True

        if tag in self.k.keys() and isinstance(self.loop_flag, bool):
            if self.k[self.tag] != 0 and tag != self.tag:
                return False
            if self.loop_flag is False:
                self._fresh_global_current_tag()
                self.loop_flag = True
            else:
                return False
        return True

    def backward(self, tag, inp, loss_fn):
        self.optimizer[tag].zero_grad()
        pre = self.model.forward(tag, inp)
        loss = loss_fn(pre, inp)
        try:
            loss.backward()
        except Exception:
            print(f'{tag} Loss Backward Error: {loss}')
            raise RuntimeError
        self.optimizer[tag].step()
        return loss, pre

    def batch_update(self, *params):
        f, inp, _ = params
        losses = {}
        pred = {}
        self.loop_flag = False if self.loop_flag is not None else None
        for tag, loss_fn in self.loss.items():
            pre, loss = None, None
            if tag in self.tags:
                if self.is_trainable(tag):
                    if self.loop_flag is None and tag in self.k.keys():
                        for idx, _ in enumerate(range(self.k[tag])):
                            loss, pre = self.backward(tag, inp, loss_fn)
                    else:
                        loss, pre = self.backward(tag, inp, loss_fn)
                    self._train_state[tag] = True

            loss = self._to_cpu(loss)
            pre = self._to_cpu(pre)
            pred.update({tag: pre})
            losses.update({tag: loss})
        return losses, pred


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True,
                 load_data_keys=('moving_t1', 'atlas', 'moving_seg'),
                 pretrain: dict = None, plot_fn=None, tags=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            load_data_keys=load_data_keys,
            plot_fn=plot_fn,
            tags=tags
        )
        self.pretrain = pretrain if pretrain is not None else dict(zip(self.tags, [0] * len(self.tags)))

    def on_epoch_start(self):
        self.model.eval()
        self.marks = {'reg': False, 'gen': False, 'inp': False}
        for k in self.pretrain.keys():
            self.pretrain[k] -= 1

    def batch_update(self, *params):
        f, inp, _ = params
        losses = {}
        pred = {}
        with torch.no_grad():
            for tag in self.tags:
                pre, loss = None, None
                pre = self.model.forward(tag, inp, train=False)
                if tag in self.loss.keys():
                    loss = self.loss[tag](pre, inp).to('cpu')
                pre = self._to_cpu(pre)
                pred.update({tag: pre})
                losses.update({tag: loss})
        return losses, pred


class TestEpoch(Epoch):
    def __init__(self, model, metrics=None, device='cpu', verbose=True, interface=None,
                 load_data_keys=('moving_t1', 'atlas', 'moving_seg'), plot_fn=None,
                 tags=None):
        """
        @param interface: used for other operations like plot images
        """
        self.desc = None
        super().__init__(
            model=model,
            loss=None,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
            load_data_keys=load_data_keys,
            plot_fn=plot_fn,
            tags=tags
        )
        self.interface = interface
        self.logs = []

    def on_epoch_start(self):
        self.model.eval()

    def set_desc(self, name):
        self.desc = name

    def batch_update(self, *params):
        f_n, inp, norm = params
        pred = {}
        with torch.no_grad():
            for tag in self.tags:
                pre = self.model.forward(tag, inp, False)
                pre = self._to_cpu(pre)
                pred.update({tag: pre})
            if self.interface is not None:
                pred = self.interface(self.desc, self.f_name, inp, pred, norm)
        return None, pred


if __name__ == '__main__':
    print('ok')
    pass
