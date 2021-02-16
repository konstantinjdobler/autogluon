import mxnet as mx
import numpy as np
from mxnet import nd

from ...task.image_classification.utils import smooth, mixup_transform



def default_train_fn(epoch, num_epochs, net, batch, batch_size, criterion, trainer, batch_fn, ctx,
                     mixup=False, label_smoothing=False, distillation=False,
                     mixup_alpha=0.2, mixup_off_epoch=0, classes=1000,
                     dtype='float32', metric=None, teacher_prob=None, train_mode=True):
    data, label = batch_fn(batch, ctx)
    if mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        if epoch >= num_epochs - mixup_off_epoch:
            lam = 1
        data = [lam * X + (1 - lam) * X[::-1] for X in data]
        if label_smoothing:
            eta = 0.1
        else:
            eta = 0.0
        label = mixup_transform(label, classes, lam, eta)
    elif label_smoothing:
        hard_label = label
        label = smooth(label, classes)

    with mx.autograd.record(train_mode=train_mode):
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
        if distillation:
            loss = [
                criterion(
                    yhat.astype('float', copy=False),
                    y.astype('float', copy=False),
                    p.astype('float', copy=False)
                )
                for yhat, y, p in zip(outputs, label, teacher_prob(data))
            ]
        else:
            loss = [criterion(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs, label)]

    for l in loss:
        l.backward()
    trainer.step(batch_size, ignore_stale_grad=True)

    if metric:
        if mixup:
            output_softmax = [
                nd.SoftmaxActivation(out.astype('float32', copy=False))
                for out in outputs
            ]
            metric.update(label, output_softmax)
        else:
            if label_smoothing:
                metric.update(hard_label, outputs)
            else:
                metric.update(label, outputs)
        return metric
    else:
        return