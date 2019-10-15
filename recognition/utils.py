import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_epoch(model, data_loader, loss_fn, optimizer, cuda):
    # train an epoch
    model.train()
    
    losses  = 0.0

    pbar = tqdm(data_loader)
    for batch_idx, (data, target) in enumerate(pbar):
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
        loss         = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses      += loss.item()

        pbar.set_description("%.5f" % loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    losses /= len(data_loader)

    return losses
    
def test_epoch(model, data_loader, loss_fn, cuda):
    # test
    with torch.no_grad():
        model.eval()
        
        losses      = 0.0
        for batch_idx, (data, target) in enumerate(data_loader):
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
                    
        losses /= len(data_loader)

        return losses

def extract_embeddings(data_loader, model, embedding_size=2048, cuda=True):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(data_loader.dataset), embedding_size))
        labels = np.zeros(len(data_loader.dataset))
        k = 0
        for images, target in data_loader:
            if cuda:
                images = images.cuda()
            embeddings[k: k+len(images)] = model.forward(images).data.cpu().numpy()
            labels[k: k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels