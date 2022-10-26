import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from Model import ProtoNet, ResNet
from Datagenerator import Datagen_test
import numpy as np
from batch_sampler import EpisodicBatchSampler
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter1d, minimum_filter
from torch.autograd import Variable
from Model import Classifier, LightModel
import os


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Adopted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    Compute the prototypes by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      bprototypes, for each one of the current classes
    '''

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    emb_dim = input_cpu.size(-1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    p = n_classes * n_support

    n_query = target.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.stack([input_cpu[idx_list] for idx_list in support_idxs])
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    num_batch = prototypes.shape[0]
    num_proto = prototypes.shape[1]

    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.cpu()[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)
    logits = -dists

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss, acc_val


def get_probability(proto_pos, neg_proto, query_set_out):
    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """

    prototypes = torch.stack([proto_pos, neg_proto]).squeeze(1)
    dists = euclidean_dist(query_set_out, prototypes)
    '''  Taking inverse distance for converting distance to probabilities'''
    logits = -dists

    prob = torch.softmax(logits, dim=1)
    inverse_dist = torch.div(1.0, dists)

    # prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:, 0]

    return prob_pos.detach().cpu().tolist()


def evaluate_prototypes_basic(conf=None, hdf_eval=None, device=None, strt_index_query=None, mean=None, std=None,
                              save_add=None):
    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = Datagen_test(hdf_eval, mean, std)
    X_pos, X_neg, X_query, hop_seg = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)

    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,
                                           batch_size=conf.eval.query_batch_size, shuffle=False)

    pos_dataset = torch.utils.data.TensorDataset(X_pos, torch.LongTensor(np.zeros(X_pos.shape[0])))
    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    neg_dataset = torch.utils.data.TensorDataset(X_neg, torch.LongTensor(np.zeros(X_neg.shape[0])))
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    if conf.train.encoder == 'Resnet':
        encoder = ResNet()
    else:
        encoder = ProtoNet()

    state_dict = torch.load(conf.path.pretrained_model, map_location=torch.device('cpu'))
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    encoder.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []

    pos_set_feat = []

    print("Creating positive prototype")
    with torch.no_grad():
        pos_set_feat = []
        for batch in tqdm(pos_loader):
            x, y = batch
            feat = encoder(x.to(device))
            pos_set_feat.append(feat)

        pos_set_feat = torch.cat(pos_set_feat, dim=0)
        pos_proto = pos_set_feat.mean(dim=0).unsqueeze(0)

        neg_set_feat = []
        for batch in tqdm(neg_loader):
            x, y = batch
            feat = encoder(x.to(device))
            neg_set_feat.append(feat)

    neg_set_feat = torch.cat(neg_set_feat, dim=0)
    neg_proto = neg_set_feat.mean(dim=0).unsqueeze(0)
    prototypes = torch.cat([pos_proto, neg_proto], dim=0).to(device)

    prob_pos_iter = []

    q_iterator = iter(q_loader)

    for batch in tqdm(q_iterator):
        x_q, y_q = batch
        x_q = x_q.to(device)
        x_query = encoder(x_q)
        y_pre = torch.softmax(-torch.cdist(x_query, prototypes), dim=1)

        probability_pos = y_pre[:, 0].detach().cpu().tolist()
        prob_pos_iter.extend(probability_pos)

    prob_final = np.array(prob_pos_iter)

    thresh = conf.eval.threshold

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > thresh, 1, 0)

    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset


def evaluate_prototypes(conf=None, hdf_eval=None, device=None, strt_index_query=None, mean=None, std=None,
                        save_add=None):
    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = Datagen_test(hdf_eval, mean, std)
    X_pos, X_neg, X_query, hop_seg = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)

    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,
                                           batch_size=conf.eval.query_batch_size, shuffle=False)

    pos_dataset = torch.utils.data.TensorDataset(X_pos, torch.LongTensor(np.zeros(X_pos.shape[0])))
    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    neg_dataset = torch.utils.data.TensorDataset(X_neg, torch.LongTensor(np.zeros(X_neg.shape[0])))
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    if conf.train.encoder == 'Resnet':
        encoder = ResNet()
    else:
        encoder = ProtoNet()

    light_state_dict = torch.load(os.path.join(conf.eval.eval_model, save_add), map_location=torch.device('cpu'))
    light = LightModel()
    light.load_state_dict(light_state_dict['light'])
    light.to(device)
    light.eval()

    state_dict = torch.load(conf.path.pretrained_model, map_location=torch.device('cpu'))
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    encoder.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []

    pos_set_feat = []

    print("Creating positive prototype")
    with torch.no_grad():
        pos_set_feat = []
        for batch in tqdm(pos_loader):
            x, y = batch
            feat = encoder(x.to(device))
            pos_set_feat.append(feat)

        pos_set_feat = torch.cat(pos_set_feat, dim=0)
        pos_proto = pos_set_feat.mean(dim=0).unsqueeze(0)

        neg_set_feat = []
        for batch in tqdm(neg_loader):
            x, y = batch
            feat = encoder(x.to(device))
            neg_set_feat.append(feat)

    neg_set_feat = torch.cat(neg_set_feat, dim=0)
    final = []
    for xxx in range(1):
        pos_indices = torch.randperm(neg_set_feat.size(0))[:]
        neg_proto = neg_set_feat[pos_indices].mean(dim=0).unsqueeze(0)
        prototypes = torch.cat([pos_proto, neg_proto], dim=0).to(device)

        prob_pos_iter = []

        q_iterator = iter(q_loader)
        with torch.no_grad():
            for batch in tqdm(q_iterator):
                x_q, y_q = batch
                x_q = x_q.to(device)
                x_query, _ = encoder(x_q, "test")
                y_pre = light(x_query)
                y_pre = torch.softmax(-torch.cdist(y_pre, prototypes), dim=1)

                probability_pos = y_pre[:, 0].detach().cpu().tolist()
                prob_pos_iter.extend(probability_pos)

        final.append(prob_pos_iter)

    prob_final = np.mean(np.array(final), axis=0)

    thresh = conf.eval.threshold

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > thresh, 1, 0)

    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset


def training_transductive_model(conf=None, hdf_eval=None, device=None, strt_index_query=None, mean=None, std=None,
                                save_add=None):
    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = Datagen_test(hdf_eval, mean, std)
    X_pos, X_neg, X_query, hop_seg = gen_eval.generate_eval()
    X_pos = torch.tensor(X_pos)

    print(X_pos.size())
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    if X_pos.size(1) <= 17:
        # X_pos = X_pos.repeat(2,1,1)
        print("The size is too small")
    else:
        max_len = X_pos.size(1)
        seg_num = max_len // 17

        def cut_x(x):
            result = []
            index = 0
            for i in range(seg_num):
                result.append(x[:, index:index + 17, :])
                index += 17

            result.append(x[:, -17:, :])
            return torch.cat(result, dim=0)

        X_pos = cut_x(X_pos)
        X_neg = cut_x(X_neg)
        X_query = cut_x(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,
                                           batch_size=conf.eval.query_batch_size, shuffle=True)
    pos_dataset = torch.utils.data.TensorDataset(X_pos, torch.LongTensor(np.zeros(X_pos.shape[0])))
    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    neg_dataset = torch.utils.data.TensorDataset(X_neg, torch.LongTensor(np.zeros(X_neg.shape[0])))
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None,
                                             batch_size=conf.eval.query_batch_size)
    train_pro = torch.load(conf.path.prototype)["prototype"].to(device)

    if conf.train.encoder == 'Resnet':
        encoder = ResNet()
    else:
        encoder = ProtoNet()

    if device == 'cpu':
        state_dict = torch.load(conf.path.pretrained_model, map_location=torch.device('cpu'))
        encoder.load_state_dict(state_dict['encoder'])

    else:
        state_dict = torch.load(conf.path.pretrained_model)
        encoder.load_state_dict(state_dict['encoder'])

    light = LightModel().to(device)

    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        pos_set_feat = []
        for batch in tqdm(pos_loader):
            x, y = batch
            feat = encoder(x.to(device))
            pos_set_feat.append(feat)

        pos_set_feat = torch.cat(pos_set_feat, dim=0)
        pos_proto = pos_set_feat.mean(dim=0).unsqueeze(0)

        neg_set_feat = []
        for batch in tqdm(neg_loader):
            x, y = batch
            feat = encoder(x.to(device))
            neg_set_feat.append(feat)

    neg_set_feat = torch.cat(neg_set_feat, dim=0)
    neg_proto = neg_set_feat.mean(dim=0).unsqueeze(0)
    prototypes = torch.cat([pos_proto, neg_proto, train_pro], dim=0).to(device)

    # classifier = Classifier([pos_proto,neg_proto]).to(device)
    # a = classifier.state_dict()
    # light.train()

    optimizer_classifier = torch.optim.Adam(light.parameters(), lr=conf.eval.lr)

    iterations = conf.eval.iterations
    p_k = conf.eval.p_k
    p_k_step = conf.eval.p_k_step
    CE = torch.nn.CrossEntropyLoss()

    for epoch in range(iterations):
        print("Epoch: {}".format(epoch))
        encoder.eval()
        light.train()
        q_iterator = iter(q_loader)
        all_hard_loss = []
        all_soft_loss = []
        all_angle_loss = []
        for batch in tqdm(q_iterator):
            neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
            x_sampled_neg = X_neg[neg_indices]
            y_sampled_neg = torch.ones(x_sampled_neg.size(0)).long()

            pos_indices = torch.randperm(len(X_pos))[::8]
            x_sampled_pos = X_pos[pos_indices]
            y_sampled_pos = torch.zeros(x_sampled_pos.size(0)).long()

            x_s = torch.cat([x_sampled_pos, x_sampled_neg], dim=0).to(device)
            y_s = torch.cat([y_sampled_pos, y_sampled_neg], dim=0).to(device)

            x_q, y_q = batch
            x_q = x_q.to(device)

            with torch.no_grad():
                f_s, f_s2 = encoder(x_s, "test")
                f_q, f_q2 = encoder(x_q, "test")
                teacher_feat = torch.cat([f_s2, f_q2], dim=0)
                teacher_vector = teacher_feat.unsqueeze(1).repeat(1,prototypes.size(0),1) - prototypes


                teacher_s = -torch.cdist(f_s2, prototypes)
                teacher_q = -torch.cdist(f_q2, prototypes)
                teacher_total = torch.cat([teacher_s, teacher_q], dim=0)

                teacher_all = torch.softmax(teacher_total / conf.eval.t, dim=1)

            student_feature_s = light(f_s)
            student_feature_q = light(f_q)
            student_feature = torch.cat([student_feature_s,student_feature_q],dim=0)
            student_vector = student_feature.unsqueeze(1).repeat(1,prototypes.size(0),1) - prototypes

            student_s = -torch.cdist(student_feature_s, prototypes)
            student_q = -torch.cdist(student_feature_q, prototypes)
            student_total = torch.cat([student_s, student_q], dim=0)

            angle_loss = -torch.mean(torch.cosine_similarity(teacher_vector,student_vector,dim=-1))



            student_all = torch.softmax(student_total, dim=1)
            y_pre_q = torch.softmax(student_q[:, :2], dim=1)

            hard_loss = CE(student_s, y_s)
            soft_loss = torch.mean(torch.sum(-teacher_all * torch.log(student_all), dim=1))

            all_hard_loss.append(hard_loss.item())
            all_soft_loss.append(soft_loss.item())
            all_angle_loss.append(angle_loss.item())
            # minimize the mutual information
            H_y = -torch.sum(torch.mean(y_pre_q, dim=0) * torch.log(torch.mean(y_pre_q, dim=0)))
            H_y_based_x = -torch.mean(torch.sum(y_pre_q * torch.log(y_pre_q), dim=1))
            loss_mutual_information = 0.85 * (soft_loss+angle_loss)/2 + 0.07 * hard_loss - 0.08 * (H_y - H_y_based_x)

            optimizer_classifier.zero_grad()
            loss_mutual_information.backward()
            optimizer_classifier.step()
        print("Hard loss : {}   Soft loss: {} Angle loss: {}".format(np.mean(all_hard_loss), np.mean(all_soft_loss), np.mean(all_angle_loss)))

        torch.save({'light': light.state_dict()},
                   os.path.join(conf.eval.eval_model, "iteration_{}_".format(epoch + 1) + save_add))
    return


def collect_train_prototype(conf, X_tr, Y_tr, device):
    if conf.train.encoder == 'Resnet':
        encoder = ResNet()
    else:
        encoder = ProtoNet()

    state_dict = torch.load(conf.path.pretrained_model, map_location=torch.device('cpu'))
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    encoder.eval()
    # print(torch.max(Y_tr)) # 45
    num_class = 46
    all_feature = [[] for i in range(num_class)]
    train_dataset = torch.utils.data.TensorDataset(X_tr, Y_tr)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=None, batch_size=1, shuffle=True)

    q_iterator = iter(train_loader)
    i = 0
    with torch.no_grad():
        for batch in tqdm(q_iterator):
            x, y = batch
            x = x.to(device)
            f = encoder(x).cpu()

            all_feature[y].append(f)
            i += 1
            if i > 5000:
                break

    all_feature_torch = []
    for i in range(num_class):
        all_feature_torch.append(torch.cat(all_feature[i], dim=0).mean(dim=0).unsqueeze(0))
    all_feature_torch = torch.cat(all_feature_torch, dim=0)
    print(all_feature_torch.shape)
    torch.save({'prototype': all_feature_torch}, os.path.join(conf.path.Model, "prototype.pth"))