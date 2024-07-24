import torch
import numpy as np
import h5py


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = torch.load("./datasets/PU1K/" + args.pu1k_data)
        if args.task == "reconstruction":
            self.data = self.data - self.data.mean(1, keepdim=True)
            self.data = self.data / self.data.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
        print("Data loaded and normalized")
        with h5py.File("./datasets/PU1K/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5", 'r') as f:
            self.data = torch.tensor(f["poisson_1024"][:])
            self.data = self.data - self.data.mean(1, keepdim=True)
            self.data = self.data / self.data.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud