import logging
import utils


def evaluate(model, loss_fn, dataloader, kd=False):
    model.eval()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    for data_batch, labels_batch in dataloader:
        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

        # Compute model output
        output_batch = model(data_batch)

        # loss is not needed in KD mode - just to speed up op
        if not kd:
            losses.update(loss_fn(output_batch, labels_batch).data,
                          data_batch.size(0))

        _, predicted = output_batch.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    loss_avg = losses.avg
    acc = 100. * correct / total

    logging.info(
        '- Eval metrics, acc:{acc:.4f}, loss: {loss_avg:.4f}'.format(acc=acc, loss_avg=loss_avg))

    my_metric = {'accuracy': acc, 'loss': loss_avg}
    return my_metric
