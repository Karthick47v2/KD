import logging

from torch.autograd import Variable
import utils


def evaluate(model, loss_fn, dataloader, kd=False):
    """Evaluate the model on the given data.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        args: (argparse.Namespace) command line arguments
        kd_mode: (bool) knowledge distillation mode

    Returns:
        my_metric: (dict) a dictionary containing evaluation metrics (accuracy and loss)
    """
    model.eval()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    for data_batch, labels_batch in dataloader:
        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # Compute model output
        output_batch = model(data_batch)

        # Compute loss (0.0 for KD mode)
        loss = 0.0 if kd else loss_fn(output_batch, labels_batch)
        losses.update(loss.data, data_batch.size(0))

        # Calculate accuracy
        _, predicted = output_batch.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    loss_avg = losses.avg
    acc = 100. * correct / total

    logging.info(
        "- Eval metrics, acc:{acc:.4f}, loss: {loss_avg:.4f}".format(acc=acc, loss_avg=loss_avg))

    my_metric = {'accuracy': acc, 'loss': loss_avg}
    return my_metric
