import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from model import MultiShapeNet, OneShapeNet
import utils
import logging


class Upsampling(object):
    def __init__(self, args):
        self.snapshot_root = f'snapshot/{args.exp_name}/'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.snapshot_root, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # logging.basicConfig(filename=os.path.join(self.snapshot_root, 'log.txt'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(str(args))
        
        self.reconstruction_dir = self.snapshot_root + "reconstruction/"
        self.upsampling_dir = self.snapshot_root + "upsampling/"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        os.makedirs(self.upsampling_dir, exist_ok=True)

    def run_pu1k_fps(self, args):
        model = MultiShapeNet(args).cuda()
        best_model = torch.load(self.snapshot_root + "models/best.pth")
        model.load_state_dict(best_model["model_state_dict"])
        
        train_loader = torch.utils.data.DataLoader(Dataset(args), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        logging.info('Reconstruction and Upsampling start!!')
        fileNames = os.listdir("datasets/PU1K/input_2048")
        model.eval()
        reconstruction_loss_buf, upsampling_loss_buf = [], []
        for index, points in enumerate(train_loader):
            points = points.cuda()
            pm = points.mean(1, keepdim=True)
            points = points - pm
            pn = points.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            points = points / pn
            pred = model(input)
            pred = pred * pn
            pred = pred + pm
            points = points * pn
            points = points + pm
            reconstruction_loss = model.get_loss(points, pred)
            pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
            np.savetxt(os.path.join(self.reconstruction_dir, fileNames[index]), pred.detach().cpu().numpy())
            
            gt = torch.from_numpy(np.loadtxt("datasets/PU1K/gt_8192/" + fileNames[index]).astype(np.float32)).cuda().unsqueeze(0)
            pred = pred.unsqueeze(0)
            fps_idx = utils.farthest_point_sample(pred, args.upsampling_point, 0)
            pred = utils.index_points(pred, fps_idx).squeeze(0)
            np.savetxt(os.path.join(self.upsampling_dir, fileNames[index]), pred.detach().cpu().numpy())
            upsampling_loss = model.get_loss(gt, pred.unsqueeze(0))

            logging.info(f'{index + 1} {fileNames[index]} Reconstruction Loss: {reconstruction_loss.detach().cpu().numpy()}')
            logging.info(f'{index + 1} {fileNames[index]} Upsampling Loss: {upsampling_loss.detach().cpu().numpy()}')
            reconstruction_loss_buf.append(reconstruction_loss.detach().cpu().numpy())
            upsampling_loss_buf.append(upsampling_loss.detach().cpu().numpy())
        logging.info(f'Average Reconstruction Loss {np.mean(reconstruction_loss_buf)}')
        logging.info(f'Average Upsampling Loss {np.mean(upsampling_loss_buf)}')
        logging.info("Reconstruction and Upsampling finish!... save results")
    
    
    def run_pu1k(self, args):
        logging.info('Reconstruction start!')
        reconstruction_loss_buf = []
        args.reconstruction_point = 2048
        model = MultiShapeNet(args).cuda()
        best_model = torch.load(self.snapshot_root + "models/best.pth")
        model.load_state_dict(best_model["model_state_dict"])
        model.eval()
        for index, file in enumerate(os.listdir("./datasets/PU1K/input_2048")):
            input = torch.from_numpy(np.loadtxt("./datasets/PU1K/input_2048/" + file).astype(np.float32)).unsqueeze(0).cuda()
            
            pm = input.mean(1, keepdim=True)
            input = input - pm
            pn = input.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            input = input / pn
            pred = model(input)
            pred = pred * pn
            pred = pred + pm
            input = input * pn
            input = input + pm
            
            reconstruction_loss = model.get_loss(pred, input)
            np.savetxt(os.path.join(self.reconstruction_dir, file), pred[0].detach().cpu().numpy())
            logging.info(f'{index + 1} {file} Reconstruction Loss: {reconstruction_loss.detach().cpu().numpy()}')
            reconstruction_loss_buf.append(reconstruction_loss.detach().cpu().numpy())
        logging.info(f'Average Reconstruction Loss {np.mean(reconstruction_loss_buf)}')
        
        logging.info("Upsampling start!")
        upsampling_loss_buf = []
        args.reconstruction_point = 8192
        model = MultiShapeNet(args).cuda()
        best_model = torch.load(self.snapshot_root + "models/best.pth")
        model.load_state_dict(best_model["model_state_dict"])
        model.eval()
        for index, file in enumerate(os.listdir("./datasets/PU1K/gt_8192")):
            input = torch.from_numpy(np.loadtxt("./datasets/PU1K/input_2048/" + file).astype(np.float32)).unsqueeze(0).cuda()
            gt = torch.from_numpy(np.loadtxt("./datasets/PU1K/gt_8192/" + file).astype(np.float32)).unsqueeze(0).cuda()
            
            pm = input.mean(1, keepdim=True)
            input = input - pm
            pn = input.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            input = input / pn
            pred = model(input)
            pred = pred * pn
            pred = pred + pm
            
            upsampling_loss = model.get_loss(pred, gt)
            np.savetxt(os.path.join(self.upsampling_dir, file), pred[0].detach().cpu().numpy())
            logging.info(f'{index + 1} {file} Upsampling Loss: {upsampling_loss.detach().cpu().numpy()}')
            upsampling_loss_buf.append(upsampling_loss.detach().cpu().numpy())
        logging.info(f'Average Upsampling Loss {np.mean(upsampling_loss_buf)}')
        
        logging.info("Reconstruction and Upsampling finish!... save results")


    def run_oneshape(self, args):
        logging.info("Reconstruction start!")
        reconstruction_loss_buf = []
        args.reconstruction_point = 2048
        for index, file in enumerate(os.listdir("./datasets/PU1K/input_2048")):
            input = torch.from_numpy(np.loadtxt("./datasets/PU1K/input_2048/" + file).astype(np.float32)).unsqueeze(0).cuda()
            
            pm = input.mean(1, keepdim=True)
            input = input - pm
            pn = input.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            input = input / pn
            
            model = OneShapeNet(args).cuda()
            best_model = torch.load(self.snapshot_root + "models/" + file + "/best.pth")    
            model.load_state_dict(best_model["model_state_dict"])
            model.eval()
            pred = model()
            
            pred = pred * pn
            pred = pred + pm
            input = input * pn
            input = input + pm
            
            reconstruction_loss = model.get_loss(pred, input)
            np.savetxt(os.path.join(self.reconstruction_dir, file), pred[0].detach().cpu().numpy())
            logging.info(f'{index + 1} {file} Restruction Loss: {reconstruction_loss.detach().cpu().numpy()}')
            reconstruction_loss_buf.append((reconstruction_loss.detach().cpu().numpy(), file))
        logging.info(f'Average Reconstruction Loss {np.mean([record[0] for record in reconstruction_loss_buf])}')
        logging.info("Reconstruction finish!... save results")
        
        logging.info("Upsampling start!")
        upsampling_loss_buf = []
        args.reconstruction_point = 8192
        for index, file in enumerate(os.listdir("./datasets/PU1K/gt_8192")):
            gt = torch.from_numpy(np.loadtxt("./datasets/PU1K/gt_8192/" + file).astype(np.float32)).unsqueeze(0).cuda()
            
            pm = input.mean(1, keepdim=True)
            input = input - pm
            pn = input.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            input = input / pn
            
            model = OneShapeNet(args).cuda()
            best_model = torch.load(self.snapshot_root + "models/" + file + "/best.pth")    
            model.load_state_dict(best_model["model_state_dict"])
            model.eval()
            pred = model()
            
            pred = pred * pn
            pred = pred + pm
            
            upsampling_loss = model.get_loss(pred, gt)
            np.savetxt(os.path.join(self.upsampling_dir, file), pred[0].detach().cpu().numpy())
            logging.info(f'{index + 1} {file} Upsampling Loss: {upsampling_loss.detach().cpu().numpy()}')
            upsampling_loss_buf.append((upsampling_loss.detach().cpu().numpy(), file))
        logging.info(f'Average Upsampling Loss {np.mean([record[0] for record in upsampling_loss_buf])}')
        logging.info("Upsampling finish!... save results")