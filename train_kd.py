import os

import utils
from tqdm import tqdm
import logging
from torch.autograd import Variable
from evaluate import evaluate
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          loss_fn_kd, warmup_scheduler, params, args):
    log_dir = os.path.join(args.model_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    teacher_model.eval()
    teacher_acc = evaluate(teacher_model, loss_fn_kd, val_dataloader, kd=True)
    print(">>>>>>>>>The teacher accuracy: {}>>>>>>>>>".format(
        teacher_acc['accuracy']))

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    for epoch in range(params.num_epochs):
        if epoch > 0:   # 0 is the warm up epoch
            scheduler.step()
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1,
                     params.num_epochs, optimizer.param_groups[0]['lr']))

        # KD Train
        train_acc, train_loss = train_kd(
            model, teacher_model, optimizer, loss_fn_kd, train_dataloader, warmup_scheduler, params, epoch)
        # Evaluate
        val_metrics = evaluate(model, loss_fn_kd, val_dataloader, kd=True)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=args.model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            file_name = "eval_best_result.json"
            best_json_path = os.path.join(args.model_dir, file_name)
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "eval_last_result.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
        # export scalar data to JSON for external processing
    writer.close()


# Defining train_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, warmup_scheduler, params, epoch):
    model.train()
    teacher_model.eval()
    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    with tqdm(total=len(dataloader)) as t:
        for train_batch, labels_batch in dataloader:
            if epoch <= 0:
                warmup_scheduler.step()

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher model
            output_teacher_batch = teacher_model(train_batch).cuda()
            output_teacher_batch = Variable(
                output_teacher_batch, requires_grad=False)

            loss = loss_fn_kd(output_batch, labels_batch,
                              output_teacher_batch, params)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            loss_avg.update(loss.data)
            losses.update(loss.item(), train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100.*correct/total
    logging.info(
        "- Train accuracy: {acc:.4f}, training loss: {loss:.4f}".format(acc=acc, loss=losses.avg))
    return acc, losses.avg


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, params, model_dir, warmup_scheduler, args):

    # dir setting, tensorboard events will save in the dirctory
    log_dir = os.path.join(args.model_dir, 'base_train')
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(params.num_epochs):
        if epoch > 0:   # 1 is the warm up epoch
            scheduler.step(epoch)

        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1,
                     params.num_epochs, optimizer.param_groups[0]['lr']))

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
            logging.info("- Found new best accuracy")
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
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            if epoch <= 0:
                warmup_scheduler.step()

            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            optimizer.zero_grad()
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            loss_avg.update(loss.data)
            losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100. * correct / total
    logging.info(
        "- Train accuracy: {acc: .4f}, training loss: {loss: .4f}".format(acc=acc, loss=losses.avg))
    return acc, losses.avg
