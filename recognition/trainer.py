import torch
from tqdm import tqdm
import torch.nn.functional as F

def fit(model, train_loader, test_loader, loss_fn, optimizer, scheduler, cuda, start_epoch=0, n_epochs=100, test_every=10):
    for epoch in range(0, start_epoch):
        scheduler.step()
        
    train_losses = []
    test_losses = []
    test_accs = []
    for epoch in range(start_epoch, start_epoch + n_epochs):
        scheduler.step()

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, cuda)
        train_losses.append(train_loss)

        if epoch % test_every == 0:
            test_loss = test_epoch(model, test_loader, loss_fn, cuda)
            test_losses.append(test_loss)

            print("Epoch %d/%d. Train loss: %.4f. Test loss: %.4f." % (epoch, start_epoch + n_epochs, train_loss, test_loss))

def train_epoch(model, train_loader, loss_fn, optimizer, cuda):
    # train an epoch
    model.train()
    
    losses = 0.0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        pbar.set_description("Processing %d/%d" % (batch_idx, len(train_loader)))

        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
        if target is not None:
            target = target.cuda()
            
        optimizer.zero_grad()
        
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
                
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses += loss.item()
        
        loss.backward()
        optimizer.step()
    
    losses /= len(train_loader)
    return losses
    
def test_epoch(model, test_loader, loss_fn, cuda):
    # test
    with torch.no_grad():
        model.eval()
        
        losses = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
            
            outputs = model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses += loss.item()
            
            # top_p, top_class = outputs[0].topk(1, dim=1)
            # equals = top_class == target[0].view(*top_class.shape)
            # accs += torch.mean(equals.type(torch.FloatTensor)).item()
        
        losses /= len(test_loader)

        return losses

def train_coteaching(train_loader, loss_fn, model1, optimizer1, model2, optimizer2,  rate_schedule, epoch, cuda):
    model1.train()
    model2.train()

    losses_1 = 0.0
    total_losses_1 = 0.0
    losses_2 = 0.0
    total_losses_2 = 0.0

    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
        if target is not None:
            target = target.cuda()

        embd_1 = model1(*data)
        embd_2 = model2(*data)

        loss_inputs = (embd_1, embd_2, target, rate_schedule[epoch - 1])
        loss_1, loss_2, total_loss_1, total_loss_2 = loss_fn(*loss_inputs)
        pbar.set_description("[Epoch %d, Kr:=%.2f] %.3f/%.3f %.3f/%.3f" \
                                % (epoch, rate_schedule[epoch - 1], loss_1.item(), total_loss_1.item(), loss_2.item(), total_loss_2.item()))

        losses_1 += loss_1.item()
        losses_2 += loss_2.item()

        total_losses_1 += total_loss_1.item()
        total_losses_2 += total_loss_2.item()

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    return losses_1 / len(train_loader), losses_2 / len(train_loader), total_losses_1 / len(train_loader), total_losses_2 /  len(train_loader)

def eval_coteaching(model1, model2, test_loader, loss_fn, cuda, metric_acc=False):
    # test
    with torch.no_grad():
        model1.eval()
        model2.eval()

        losses_1 = 0.0
        losses_2 = 0.0
        accs_1 = 0.0
        accs_2 = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

            embd_1 = model1(*data)
            embd_2 = model2(*data)

            loss_inputs = (embd_1, embd_2, target, 1.0)
            loss_1, loss_2, _, _ = loss_fn(*loss_inputs)

            losses_1 += loss_1.item()
            losses_2 += loss_2.item()

            if metric_acc:
                prec_1, _ = accuracy(embd_1, target, topk=(1, 2))
                accs_1 += prec_1
                
                prec_2, _ = accuracy(embd_2, target, topk=(1, 2))
                accs_2 += prec_2

        return losses_1 / len(test_loader), losses_2 / len(test_loader), accs_1 / len(test_loader),  accs_2 / len(test_loader)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res