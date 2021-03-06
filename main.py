import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
import os

def load_weights(filepath):
    path = os.path.join(filepath)
    model = torch.load(path + '_model.pt')
    state = torch.load(path + '_model_state_dict.pt')
    model.load_state_dict(state)
    print('Loading weights from {}'.format(filepath))
    return model

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    if args.dataset == 'cifar10':
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(0, auc))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
        center = torch.FloatTensor(feature_space).mean(dim=0)
        if args.angular:
            center = F.normalize(center, dim=-1)
        center = center.to(device)
        for epoch in range(args.epochs):
            running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
            print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
            auc, _ = get_score(model, device, train_loader, test_loader)
            print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    elif args.dataset == 'mvtec':
        auc, feature_space = get_score_mvtec(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(0, auc))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
        center = torch.FloatTensor(feature_space).mean(dim=0)
        if args.angular:
            center = F.normalize(center, dim=-1)
        center = center.to(device)
        for epoch in range(args.epochs):
            running_loss = run_epoch_mvtec(model, train_loader_1, optimizer, center, device, args.angular)
            print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
            auc, _ = get_score_mvtec(model, device, train_loader, test_loader)
            print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    else:
        print("Unsupported dataset! \n")
        exit()
    torch.save(model, args.filepath + '_model.pt')
    torch.save(model.state_dict(), args.filepath + '_model_state_dict.pt')

def test_model(model, train_loader, test_loader, device, args):
    model.eval()
    if args.dataset == 'mvtec':
        auc, _ = get_score_mvtec(model, device, train_loader, test_loader)
    elif args.dataset == 'cifar10':
        auc, _ = get_score(model, device, train_loader, test_loader)
    else: 
        print("Unsupported dataset! \n")
        exit()
    print('Epoch: {}, AUROC is: {}'.format(0, auc))

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):
        import pdb; pdb.set_trace()
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def run_epoch_mvtec(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _, _) in tqdm(train_loader, desc='Train...'):
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def get_score_mvtec(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device) # (64, 3, 224, 224)
            features = model(imgs) # (64, 2048)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0) # (192, 2048)

    test_feature_space = []
    test_labels = []
    gt_mask_list = []
    score_map_list = []
    with torch.no_grad():
        for (imgs, labels, mask) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
            # gt_mask_list.extend(mask.cpu().detach().numpy())
        test_feature_space = torch.cat(test_feature_space, dim=0) # (132, 2048)
        test_labels = torch.cat(test_labels, dim=0) # (132,)
        
        # for t_idx in tqdm(range(test_feature_space.shape[0]), desc='MVTec Localization'):
        #     feat_map = train_feature_space[t_idx].unsqueeze(-1)
        #     test_map = test_feature_space[t_idx]
        #     for d_idx in range(feat_map.shape[0]):
        #         dist_matrix = torch.pairwise_distance(feat_map[d_idx:], test_map)
        #         dist_matrix = F.interpolate(dist_matrix.unsqueeze(0).unsqueeze(0), size=224*224,
        #                                   mode='linear', align_corners=False) 
        #         score_map_list.append(dist_matrix)
        #     # dist_matrix = torch.cat(dist_matrix_list, 0)
            

    train_feature_space = train_feature_space.contiguous().cpu().numpy()
    test_feature_space = test_feature_space.contiguous().cpu().numpy() # (132, 2048)
    test_labels = test_labels.cpu().numpy()
    # flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
    # flatten_score_map_list = np.concatenate(score_map_list).ravel()

    distances = utils.knn_score(train_feature_space, test_feature_space) # (132, )

    auc = roc_auc_score(test_labels, distances)
    # per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'cifar10':
        filepath = args.load_path + str(args.backbone)+'_'+str(args.dataset)+'_'+str(args.label)
    else: # mvtec
        filepath = args.load_path + str(args.backbone)+'_'+str(args.dataset)+'_'+str(args.class_name)
    print(device)
    train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone, args=args)

    if args.mode == 'train':
        model = utils.Model(args.backbone)
        model = model.to(device)    
        args.filepath = filepath
        os.makedirs(args.save_path, exist_ok=True)
        train_model(model, train_loader, test_loader, train_loader_1, device, args)

    elif args.mode == 'test':
        model = load_weights(filepath)
        model = model.to(device)
        test_model(model, train_loader, test_loader, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10', type=str, help='mvtec/cifar10 (Default=cifar10)')
    parser.add_argument('--class_name', '-cl', default='capsule', type=str, help='class name for mvtec')
    parser.add_argument('--data_path', default='/home/juyeon/data/mvtec', type=str)
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default=152, type=int, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--gpu', default='0', type=str, help='gpu number')
    parser.add_argument('--mode', default='train', type=str, help='train/test mode')
    parser.add_argument('--save_path', default='./models/', type=str, help='where to save the weights')
    parser.add_argument('--load_path', default='./models/', type=str, help='where to get the weights')

    args = parser.parse_args()
    main(args)
