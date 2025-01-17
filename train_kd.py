import os
import torch
import utils
from tqdm import tqdm
import logging

import numpy as np

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          loss_fn_kd, warmup_scheduler, params, args):
    log_dir = os.path.join(args.model_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    teacher_model.eval()
    # teacher_acc = evaluate(teacher_model, loss_fn_kd, val_dataloader, kd=True)

    # logging.info(
    #     f'>>>>>>>>>The teacher accuracy: {teacher_acc["accuracy"]}>>>>>>>>>')

    cm = np.ones((args.num_class, args.num_class))

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    for epoch in range(params['num_epochs']):
        if epoch > 0:   # 0 - warm up epoch
            scheduler.step()
        logging.info(
            f'Epoch {epoch + 1}/{params["num_epochs"]}, lr:{optimizer.param_groups[0]["lr"]}')

        train_acc, train_loss, cm = train_kd(
            model, teacher_model, optimizer, loss_fn_kd, train_dataloader, warmup_scheduler, params, epoch, cm)
        val_metrics = evaluate(
            model, loss_fn_kd, val_dataloader, kd=True)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=args.model_dir)

        if is_best:
            best_val_acc = val_acc
            file_name = "eval_best_result.json"
            best_json_path = os.path.join(args.model_dir, file_name)
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(args.model_dir, "eval_last_result.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
    writer.close()


def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, warmup_scheduler, params, epoch, cm):
    model.train()
    teacher_model.eval()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    labels = []
    preds = []

    with tqdm(total=len(dataloader)) as t:
        for train_batch, labels_batch in dataloader:
            if epoch <= 0:
                warmup_scheduler.step()

            train_batch, labels_batch = train_batch.to(
                device), labels_batch.to(device)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            pred = output_batch.argmax(1)
            preds += pred.cpu()
            labels += labels_batch.cpu()

            # get one batch output from teacher model
            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch).to(device)

                # with CM
                # for idx in range(len(labels_batch)):
                #     pred = output_teacher_batch[idx].argmax(0)
                #     if pred != labels_batch[idx]:
                #         temp = torch.tensor(cm[labels_batch[idx]], device=device, dtype=torch.float32)
                #         temp /= sum(temp)

                #         output_teacher_batch[idx] = torch.where(output_teacher_batch[idx] < 0,
                #                                               output_teacher_batch[idx] / temp,
                #                                               output_teacher_batch[idx])

                #         output_teacher_batch[idx] = torch.where(output_teacher_batch[idx] > 0,
                #                                               output_teacher_batch[idx] * temp,
                #                                               output_teacher_batch[idx])

                # with PS
                # idx_p = torch.argmax(output_teacher_batch[idx], dim=0)

                # temp = output_teacher_batch[idx][idx_p]
                # output_teacher_batch[idx][idx_p] = output_teacher_batch[idx][labels_batch[idx]]
                # output_teacher_batch[idx][labels_batch[idx]] = temp

            loss = loss_fn_kd(output_batch, labels_batch,
                              output_teacher_batch, params)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            predicted = output_batch.argmax(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            loss_avg.update(loss.data)
            losses.update(loss.item(), train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100. * correct/total
    logging.info(
        '- Train accuracy: {acc:.4f}, training loss: {loss:.4f}'.format(acc=acc, loss=losses.avg))
    return acc, losses.avg, utils.calc_cm(labels, preds, [x for x in range(0, len(cm[0]))])


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, params, model_dir, warmup_scheduler, args):
    log_dir = os.path.join(args.model_dir, 'base_train')
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(params['num_epochs']):
        if epoch > 0:   # 0 - warm up epoch
            scheduler.step(epoch)

        logging.info(
            f'Epoch {epoch + 1}/{params["num_epochs"]}, lr:{optimizer.param_groups[0]["lr"]}')

        train_acc, train_loss = train(
            model, optimizer, loss_fn, train_dataloader, epoch, warmup_scheduler)

        val_metrics = evaluate(model, loss_fn, val_dataloader)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        if is_best:
            best_val_acc = val_acc
            best_json_path = os.path.join(model_dir, "eval_best_results.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "eval_last_results.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
    writer.close()


def train(model, optimizer, loss_fn, dataloader, epoch, warmup_scheduler):
    model.train()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    with tqdm(total=len(dataloader)) as t:
        for train_batch, labels_batch in dataloader:
            train_batch, labels_batch = train_batch.to(
                device), labels_batch.to(device)
            if epoch <= 0:
                warmup_scheduler.step()

            optimizer.zero_grad()
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()

            predicted = output_batch.argmax(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            loss_avg.update(loss.data)
            losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100. * correct / total
    logging.info(
        '- Train accuracy: {acc: .4f}, training loss: {loss: .4f}'.format(acc=acc, loss=losses.avg))
    return acc, losses.avg


def evaluate(model, loss_fn, dataloader, kd=False):
    model.eval()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            data_batch, labels_batch = data_batch.to(
                device), labels_batch.to(device)

            output_batch = model(data_batch)

            # loss is not needed in KD mode - just to speed up op
            if not kd:
                losses.update(loss_fn(output_batch, labels_batch).data,
                              data_batch.size(0))

            predicted = output_batch.argmax(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

    loss_avg = losses.avg
    acc = 100. * correct / total

    logging.info(
        '- Eval metrics, acc:{acc:.4f}, loss: {loss_avg:.4f}'.format(acc=acc, loss_avg=loss_avg))

    return {'accuracy': acc, 'loss': loss_avg}
