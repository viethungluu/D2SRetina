import numpy as np

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

def pdist(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx