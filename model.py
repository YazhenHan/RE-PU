import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools
from vector_quantize_pytorch import VectorQuantize


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)

    return x

def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)      # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                       # (batch_size, num_points, k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
    return feature                              # (batch_size, 2*num_dims, num_points, k)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, index):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.tensor([index], dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()

    fps_idx = farthest_point_sample(xyz, npoint, torch.randint(0, N, (1))).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class DGCNN_Cls_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Cls_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.task = args.task
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.feat_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x):        
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)
        x0 = self.conv5(x)                      # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]    # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)                   # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
        if self.task == 'classify':
            return feat, x0
        elif self.task == 'reconstruct':
            return feat                         # (batch_size, 1, feat_dims)


class Point_Transform_Net(nn.Module):
    def __init__(self):
        super(Point_Transform_Net, self).__init__()
        self.k = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)
        return x                                # (batch_size, 3, 3)


class DGCNN_Seg_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Seg_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.transform_net = Point_Transform_Net()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.feat_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)
        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feat = x.unsqueeze(1)                   # (batch_size, num_points) -> (batch_size, 1, emb_dims)
        return feat                             # (batch_size, 1, emb_dims)


class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048   # input point cloud size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )
    def graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x
    def forward(self, pts):
        pts = pts.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)                 # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat                             # (batch_size, 1, feat_dims)


class PointNet_Encoder(nn.Module):
    def __init__(self):
        super(PointNet_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.encoder(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.transpose(2, 1)
        return x


class OA_Layer(nn.Module):
    def __init__(self):
        super(OA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(256, 256 // 4, 1)
        self.k_conv = nn.Conv1d(256, 256 // 4, 1)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.softmax = nn.Softmax(dim=-1)
        self.v_conv = nn.Conv1d(256, 256, 1)
        self.trans_conv = nn.Conv1d(256, 256, 1)
        self.after_norm = nn.BatchNorm1d(256)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x, xyz):
        x = x + xyz
        q = self.q_conv(x).permute(0, 2, 1)
        k = self.k_conv(x)
        v = self.v_conv(x)
        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
    
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class OffsetAttention_Encoder(nn.Module):
    def __init__(self):
        super(OffsetAttention_Encoder, self).__init__()
        self.mlp = nn.Conv1d(3, 256, 1)
        self.oa1 = OA_Layer()
        self.oa2 = OA_Layer()
        self.oa3 = OA_Layer()
        self.oa4 = OA_Layer()
        self.oa_dc = nn.Sequential(nn.Conv1d(256 * 4, 512, 1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        p1 = self.oa1(x)
        p2 = self.oa2(p1)
        p3 = self.oa3(p2)
        p4 = self.oa4(p3)
        p = torch.cat((p1, p2, p3, p4), dim=1)
        points = self.oa_dc(p)
        lc = torch.max(points, 2, keepdim=True)[0]
        return lc.transpose(1, 2)


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = OA_Layer()
        self.sa2 = OA_Layer()
        self.sa3 = OA_Layer()
        self.sa4 = OA_Layer()

    def forward(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PCT_Encoder(nn.Module):
    def __init__(self, args):
        super(PCT_Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.gather_local_0 = Local_op(128, 128)
        self.gather_local_1 = Local_op(256, 256)
        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                  nn.BatchNorm1d(512))
        
    def forward(self, x):
        xyz = x
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(512, 0.1, 32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(256, 0.2, 32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)
        x = self.pt_last(feature_1, new_xyz)
        x =  torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(x.shape[0], -1)
        x = self.mlp2(x).unsequeeze(1)
        return x


class Oneshape_MLP_Decoder(nn.Module):
    def __init__(self, args):
        super(Oneshape_MLP_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        self.decoder = nn.Sequential(
            nn.Conv1d(d, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
            nn.BatchNorm1d(3),
            nn.Tanh()
        )
            
    def forward(self, prior):
        points = self.decoder(prior)
        return points.transpose(1, 2) 


class Oneshape_FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(Oneshape_FoldNet_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        self.folding1 = nn.Sequential(
            nn.Conv1d(d, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
            nn.BatchNorm1d(3),
            nn.Tanh()
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(3, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
            nn.BatchNorm1d(3),
            nn.Tanh()
        )

    def forward(self, prior):
        folding_result1 = self.folding1(prior)         
        folding_result2 = self.folding2(folding_result1)  
        return folding_result2.transpose(1, 2) 


class Oneshape_OffsetAttention_Decoder(nn.Module):
    def __init__(self, args):
        super(Oneshape_OffsetAttention_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        self.mlp = nn.Conv1d(d, 256, 1)
        self.oa1 = OA_Layer()
        self.oa2 = OA_Layer()
        self.oa3 = OA_Layer()
        self.oa4 = OA_Layer()
        self.oa_dc = nn.Sequential(nn.Conv1d(256 * 4, 256, 1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv1d(256, 3, 1),
                                    nn.BatchNorm1d(3),
                                    nn.Tanh())

    def forward(self, prior):
        p = self.mlp(prior)
        p1 = self.oa1(p)
        p2 = self.oa2(p1)
        p3 = self.oa3(p2)
        p4 = self.oa4(p3)
        p = torch.cat((p1, p2, p3, p4), dim=1)
        points = self.oa_dc(p)
        return points.transpose(1, 2)


class Multishape_MLP_Decoder(nn.Module):
    def __init__(self, args):
        super(Multishape_MLP_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        d = args.feat_dims + d
        self.decoder = nn.Sequential(
            nn.Conv1d(d, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
            nn.BatchNorm1d(3),
            nn.Tanh()
        )

    def forward(self, feature, prior):
        feature = feature.transpose(1, 2).repeat(1, 1, self.reconstruction_point)
        cat1 = torch.cat((feature, prior), dim=1) 
        points = self.decoder(cat1)
        return points.transpose(1, 2) 


class Multishape_FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(Multishape_FoldNet_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        d = args.feat_dims + d
        self.folding1 = nn.Sequential(
            nn.Conv1d(d, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
            nn.BatchNorm1d(3)
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims + 3, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.BatchNorm1d(args.feat_dims),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
            nn.BatchNorm1d(3)
        )

    def forward(self, feature, prior):
        feature = feature.transpose(1, 2).repeat(1, 1, self.reconstruction_point)
        # points = self.build_grid(x.shape[0]).transpose(1, 2).to(x.device)
        cat1 = torch.cat((feature, prior), dim=1) 
        folding_result1 = self.folding1(cat1)         
        cat2 = torch.cat((feature, folding_result1), dim=1) 
        folding_result2 = self.folding2(cat2)  
        return folding_result2.transpose(1, 2) 


class Multishape_OffsetAttention_Decoder(nn.Module):
    def __init__(self, args):
        super(Multishape_OffsetAttention_Decoder, self).__init__()
        d = 3 if args.prior.startswith('sphere') else 2
        d = args.feat_dims + d
        self.mlp = nn.Conv1d(d, 256, 1)
        self.oa1 = OA_Layer()
        self.oa2 = OA_Layer()
        self.oa3 = OA_Layer()
        self.oa4 = OA_Layer()
        self.oa_dc = nn.Sequential(nn.Conv1d(256 * 4, 256, 1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv1d(256, 3, 1),
                                    nn.BatchNorm1d(3),
                                    nn.Tanh())

    def forward(self, feature, prior):
        feature = feature.transpose(1, 2).repeat(1, 1, prior.shape[2])
        p = self.mlp(torch.cat((feature, prior), dim=1))
        p1 = self.oa1(p)
        p2 = self.oa2(p1)
        p3 = self.oa3(p2)
        p4 = self.oa4(p3)
        p = torch.cat((p1, p2, p3, p4), dim=1)
        points = self.oa_dc(p)
        return points.transpose(1, 2)


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def batch_pairwise_dist(self, x, y):
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        return loss_1 + loss_2


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, preds, gts):
        return torch.mean((preds - gts).pow(2))


def build_encoder(args): # [B, 1, C]
    if args.encoder == 'pointnet':
        return PointNet_Encoder()
    elif args.encoder == 'foldnet':
        return FoldNet_Encoder(args)
    elif args.encoder == 'dgcnn_cls':
        return DGCNN_Cls_Encoder(args)
    elif args.encoder == 'dgcnn_seg':
        return DGCNN_Seg_Encoder(args)
    elif args.encoder == 'oa':
        return OffsetAttention_Encoder()
    elif args.encoder == 'pct':
        return PCT_Encoder(args)


def build_prior(args): # [B, C, N]
    if args.prior == 'square_grid':
        grid = int(args.reconstruction_point ** 0.5)
        meshgrid = [[-1.0, 1.0, grid], [-1.0, 1.0, grid]]
        x = np.linspace(*meshgrid[0])
        y = np.linspace(*meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
    elif args.prior == 'square_fib':
        phi = (1 + np.sqrt(5)) / 2  
        x = (np.arange(1, args.reconstruction_point + 1) / phi) % 1  
        y = np.arange(1, args.reconstruction_point + 1) / args.reconstruction_point
        points = np.stack((x * 2 - 1, y * 2 - 1), axis=-1)
    elif args.prior == 'square_ham':
        points = np.zeros((args.reconstruction_point, 2))
        for i in range(args.reconstruction_point):
            points[i, 0] = i / args.reconstruction_point
            result = 0
            f = 1 / 2
            j = i
            while j > 0:
                result += (j % 2) * f
                j //= 2
                f /= 2
            points[i, 1] = result
    elif args.prior == 'square_uniform':
        points = np.random.rand(args.reconstruction_point, 2)
        points = points * 2 - 1
    elif args.prior == 'square_gaussian':
        points = np.random.randn(args.reconstruction_point, 2)
        
    elif args.prior == 'disk_fib':
        phi = (1 + np.sqrt(5)) / 2  
        x = (np.arange(1, args.reconstruction_point + 1) / phi) % 1  
        y = np.arange(1, args.reconstruction_point + 1) / args.reconstruction_point
        theta = 2 * np.pi * x  
        r = np.sqrt(y)  
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.stack((x, y), axis=-1)
        
    elif args.prior == 'sphere_fib':
        phi = (1 + np.sqrt(5)) / 2  
        x = (np.arange(1, args.reconstruction_point + 1) / phi) % 1  
        y = np.arange(1, args.reconstruction_point + 1) / args.reconstruction_point
        theta = 2 * np.pi * x  
        phi = np.arccos(1 - 2 * y)  
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.stack((x, y, z), axis=-1)
    elif args.prior == 'sphere_uniform':
        points = np.random.randn(args.reconstruction_point, 3)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    elif args.prior == 'sphere_gaussian':
        points = np.random.randn(args.reconstruction_point, 3)
        
    if args.add_noise:
        points += np.random.uniform(-0.02, 0.02, points.shape)
    points = np.repeat(points[np.newaxis, ...], repeats=args.batch_size, axis=0)
    points = torch.tensor(points)
    points = points.transpose(2, 1).cuda()
    return points.float()


def build_decoder(args): # [B, N, 3]
    if args.mode == 'oneshape':
        if args.decoder == 'foldnet':
            return Oneshape_FoldNet_Decoder(args)
        elif args.decoder == 'mlp':
            return Oneshape_MLP_Decoder(args)
        elif args.decoder == 'oa':
            return Oneshape_OffsetAttention_Decoder(args)
    elif args.mode == 'multishape':
        if args.decoder == 'foldnet':
            return Multishape_FoldNet_Decoder(args)
        elif args.decoder == 'mlp':
            return Multishape_MLP_Decoder(args)
        elif args.decoder == 'oa':
            return Multishape_OffsetAttention_Decoder(args)


def build_loss(args):
    if args.loss == 'chamfer':
        return ChamferLoss()
    elif args.loss == 'l2':
        return L2Loss()
    

class OneShapeNet(nn.Module):
    def __init__(self, args):
        super(OneShapeNet, self).__init__()
        self.prior = build_prior(args)
        self.decoder = build_decoder(args)
        self.loss = build_loss(args)
    
    def setPrior(self, args):
        self.prior = build_prior(args)

    def forward(self):
        output = self.decoder(self.prior)
        return output

    def get_loss(self, input, output):
        return self.loss(input, output)
    

class MultiShapeNet(nn.Module):
    def __init__(self, args):
        super(MultiShapeNet, self).__init__()
        self.encoder = build_encoder(args)
        self.prior = build_prior(args)
        self.decoder = build_decoder(args)
        self.loss = build_loss(args)

    def forward(self, input):
        feature = self.encoder(input)
        if self.vq:
            vq = VectorQuantize(
                dim = 512,
                codebook_size = 127,     # codebook size
                decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.   # the weight on the commitment loss
            )
            feature, indices, commit_loss = vq(feature) # (1, 1024, 256), (1, 1024), (1)
            self.commit_loss = commit_loss
        output = self.decoder(feature, self.prior)
        return output

    def get_loss(self, input, output):
        if self.vq:
            return self.loss(input, output) + self.commit_loss
        else:
            return self.loss(input, output)
    
    
class MultiShapeVQNet(nn.Module):
    def __init__(self, args):
        super(MultiShapeNet, self).__init__()
        self.encoder = build_encoder(args)
        
        self.vq_embedding = nn.Embedding(128, 512)
        self.vq_embedding.weight.data.uniform_(-1.0 / 128,
                                               1.0 / 128)

        self.prior = build_prior(args)
        self.decoder = build_decoder(args)
        self.loss = build_loss(args)
        
    def forward(self, x):
        # encode
        ze = self.encoder(x)
        
        ze = ze.permute(0, 2, 1)
        # ze: [N, C, H, W] [B, C, 1]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        B, C, N = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1)
        ze_broadcast = ze.reshape(B, 1, C, N)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 2, 1)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()
        decoder_input = decoder_input.permute(0, 2, 1)
        
        # decode
        x_hat = self.decoder(decoder_input, self.prior)
        
        return x_hat, ze, zq

    def get_loss(self, input, output):
        return self.loss(input, output)