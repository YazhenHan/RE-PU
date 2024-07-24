import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import MultiShapeNet, OneShapeNet, MultiShapeVQNet
from dataset import Dataset
import utils
import logging


class Reconstruction(object):
    def __init__(self, args):
        self.snapshot_root = f'snapshot/{args.exp_name}/' 
        
        if args.model_path == '':
            if not os.path.exists(self.snapshot_root):
                os.makedirs(self.snapshot_root)
                os.makedirs(self.snapshot_root + "models", exist_ok=True)
            else:
                choose = input("Remove " + self.snapshot_root + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(self.snapshot_root)
                    os.makedirs(self.snapshot_root)
                    os.makedirs(self.snapshot_root + "models", exist_ok=True)
                else:
                    sys.exit(0)
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

        logging.info("Copying codes")
        os.makedirs(self.snapshot_root + "/codes", exist_ok=True)
        shutil.copy("main.py", self.snapshot_root + "/codes")
        shutil.copy("dataset.py", self.snapshot_root + "/codes")
        shutil.copy("model.py", self.snapshot_root + "/codes")
        shutil.copy("loss.py", self.snapshot_root + "/codes")
        shutil.copy("reconstruction.py", self.snapshot_root + "/codes")
        shutil.copy("upsampling.py", self.snapshot_root + "/codes")
        shutil.copy("utils.py", self.snapshot_root + "/codes")


    def run_pu1k(self, args):
        train_loader = torch.utils.data.DataLoader(Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        logging.info(f"Training set size: {train_loader.dataset.__len__()}")
        if args.vq:
            model = MultiShapeVQNet(args).cuda()
        else:
            model = MultiShapeNet(args).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)
        writer = SummaryWriter(log_dir=self.snapshot_root)
        
        if args.model_path != '':
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
        else:
            start_epoch = 0
            best_loss = 1.0
        logging.info('Training start!!')
        model.train()
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            rec_buf, emb_buf, com_buf = [], [], []
            loss_buf = []
            for pts in tqdm(train_loader):
                pts = pts.cuda()
                # forward
                if args.vq:
                    pred, ze, zq = model(pts)
                    rec_loss = model.get_loss(pts, pred)
                    emb_loss = 0.001 * nn.MSELoss()(ze.detach(), zq)
                    com_loss = 0.001 * nn.MSELoss()(ze, zq.detach())
                    rec_buf.append(rec_loss.detach().cpu().numpy())
                    emb_buf.append(emb_loss.detach().cpu().numpy())
                    com_buf.append(com_loss.detach().cpu().numpy())
                    loss = rec_loss + emb_loss + 0.25 * com_loss
                else:
                    pred = model(pts)
                    loss = model.get_loss(pts, pred)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_buf.append(loss.detach().cpu().numpy())
            # finish one epoch
            epoch_time = time.time() - epoch_start_time
            if args.vq:
                rec_loss = np.mean(rec_buf)
                emb_loss = np.mean(emb_buf)
                com_loss = np.mean(com_buf)
                loss = rec_loss + emb_loss + 0.25 * com_loss
                logging.info(f'Epoch {epoch + 1}: Rec Loss {rec_loss}, Emb Loss {emb_loss}, Com Loss {com_loss}, Loss {loss}, time {epoch_time: .4f}s')
                writer.add_scalar('Rec_Loss', rec_loss, epoch + 1)
                writer.add_scalar('Emb_Loss', emb_loss, epoch + 1)
                writer.add_scalar('Com_Loss', com_loss, epoch + 1)
            loss = np.mean(loss_buf)
            logging.info(f'Epoch {epoch + 1}: Loss {loss}, time {epoch_time: .4f}s')
            writer.add_scalar('Train_Loss', loss, epoch + 1)

            if (epoch + 1) % args.snapshot_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                }, f"{self.snapshot_root}models/checkpoint.pth")
                logging.info(f"Loss: {loss}, Save model to {self.snapshot_root}models/{args.pu1k_data}_checkpoint.pth")
                # print(f"Loss: {loss}, Save model to {self.snapshot_root}models/{args.pu1k_data}_{epoch + 1}.pth")

            if loss < best_loss:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                }, f"{self.snapshot_root}models/best.pth")
                best_loss = loss
                writer.add_scalar('Best_Loss', best_loss, epoch + 1)
                logging.info(f"Loss: {best_loss}, Save model to {self.snapshot_root}models/{args.pu1k_data}_best.pth")
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.flush()
        writer.close()
        logging.info("Training finish!... save training results")


    def run_oneshape(self, args):
        logging.info("Training start!")
        for index, file in enumerate(os.listdir("./datasets/PU1K/input_2048")):
            os.mkdir(self.snapshot_root + "models/" + file)
            points = torch.from_numpy(np.loadtxt("./datasets/PU1K/input_2048/" + file).astype(np.float32)).unsqueeze(0).cuda()
            points = points - points.mean(1, keepdim=True)
            points = points / points.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            
            model = OneShapeNet(args).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001*16/1, betas=(0.9, 0.999), weight_decay=1e-6)
            writer = SummaryWriter(log_dir=self.snapshot_root + "models/" + file)
            
            if args.model_path != '':
                checkpoint = torch.load(args.model_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_loss = checkpoint["best_loss"]
            else:
                start_epoch = 0
                best_loss = 1.0
            logging.info(f"{index + 1} {file} Training start!")
            reconstruction_point = 1000
            for epoch in tqdm(range(start_epoch, args.epochs)):
                model.train()
                if args.progressive:
                    args.reconstruction_point = reconstruction_point
                    model.setPrior(args)
                    reconstruction_point = max((reconstruction_point + 1) % 2025, 1000)
                    fps_idx = utils.farthest_point_sample(points, reconstruction_point, 0)
                    pts = utils.index_points(points, fps_idx)
                    writer.add_scalar('reconstruction_point', reconstruction_point, epoch + 1)
                pred = model()
                loss = model.get_loss(pts, pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    args.reconstruction_point = 2048
                    model.setPrior(args)
                    pred = model()
                    loss = model.get_loss(points, pred)
                    
                if loss < best_loss:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    }, f"{self.snapshot_root}models/{file}/best.pth")
                    best_loss = loss
                    writer.add_scalar('Best_Loss', best_loss, epoch + 1)
                    # print(f"Loss: {best_loss}, Save model to {self.snapshot_root}models/{file}/best.pth")

                if (epoch + 1) % args.snapshot_interval == 0:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    }, f"{self.snapshot_root}models/{file}/checkpoint.pth")
                    # print(f"Loss: {loss}, Save model to {self.snapshot_root}models/{file}/checkpoint.pth")

                writer.add_scalar('Train_Loss', loss, epoch + 1)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
                writer.flush()
            writer.close()
            logging.info(f"Loss: {best_loss}, Save model to {self.snapshot_root}models/{file}/best.pth")
            logging.info(f"{index + 1} {file} Training finish!... save training results")
        logging.info("Training finish!... save training results")


    def run_oneshape_progressive(self, args):
        logging.info("Training start!")
        for index, file in enumerate(os.listdir("./datasets/PU1K/progressive/")):
            os.mkdir(self.snapshot_root + "models/" + file)
            
            model = OneShapeNet(args).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001*16/1, betas=(0.9, 0.999), weight_decay=1e-6)
            writer = SummaryWriter(log_dir=self.snapshot_root + "models/" + file)
            
            if args.model_path != '':
                checkpoint = torch.load(args.model_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_loss = checkpoint["best_loss"]
            else:
                start_epoch = 0
                best_loss = 1.0
            logging.info(f"{index + 1} {file} Training start!")
            for f in os.listdir("./datasets/PU1K/progressive/" + file):
                points = torch.from_numpy(np.loadtxt("./datasets/PU1K/progressive/" + file + "/" + f).astype(np.float32)).unsqueeze(0).cuda()
                points = points - points.mean(1, keepdim=True)
                points = points / points.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
                
                reconstruction_point = 1000
                for epoch in tqdm(range(start_epoch, args.epochs)):
                    model.train()
                    if args.progressive:
                        args.reconstruction_point = reconstruction_point
                        model.setPrior(args)
                        reconstruction_point = max((reconstruction_point + 1) % 2025, 1000)
                        writer.add_scalar('reconstruction_point', reconstruction_point, epoch + 1)
                    pred = model()
                    loss = model.get_loss(points, pred)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        args.reconstruction_point = 2025
                        model.setPrior(args)
                        output = model()
                        loss = model.get_loss(points, output)
                        
                    if loss < best_loss:
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_loss": best_loss,
                        }, f"{self.snapshot_root}models/{file}/best.pth")
                        best_loss = loss
                        writer.add_scalar('Best_Loss', best_loss, epoch + 1)
                        # print(f"Loss: {best_loss}, Save model to {self.snapshot
                        
            points = torch.from_numpy(np.loadtxt("./datasets/PU1K/input_2048/" + file).astype(np.float32)).unsqueeze(0).cuda()
            points = points - points.mean(1, keepdim=True)
            points = points / points.norm(2, 2, keepdim=True).max(dim=1, keepdim=True)[0]
            
            model = OneShapeNet(args).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001*16/1, betas=(0.9, 0.999), weight_decay=1e-6)
            writer = SummaryWriter(log_dir=self.snapshot_root + "models/" + file)
            
            if args.model_path != '':
                checkpoint = torch.load(args.model_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_loss = checkpoint["best_loss"]
            else:
                start_epoch = 0
                best_loss = 1.0
            logging.info(f"{index + 1} {file} Training start!")
            reconstruction_point = 1000
            for epoch in tqdm(range(start_epoch, args.epochs)):
                model.train()
                if args.progressive:
                    args.reconstruction_point = reconstruction_point
                    model.setPrior(args)
                    reconstruction_point = max((reconstruction_point + 1) % 2025, 1000)
                    writer.add_scalar('reconstruction_point', reconstruction_point, epoch + 1)
                pred = model()
                loss = model.get_loss(points, pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    args.reconstruction_point = 2025
                    model.setPrior(args)
                    output = model()
                    loss = model.get_loss(points, output)
                    
                if loss < best_loss:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    }, f"{self.snapshot_root}models/{file}/best.pth")
                    best_loss = loss
                    writer.add_scalar('Best_Loss', best_loss, epoch + 1)
                    # print(f"Loss: {best_loss}, Save model to {self.snapshot_root}models/{file}/best.pth")

                if (epoch + 1) % args.snapshot_interval == 0:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    }, f"{self.snapshot_root}models/{file}/checkpoint.pth")
                    # print(f"Loss: {loss}, Save model to {self.snapshot_root}models/{file}/checkpoint.pth")

                writer.add_scalar('Train_Loss', loss, epoch + 1)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
                writer.flush()
            writer.close()
            logging.info(f"Loss: {best_loss}, Save model to {self.snapshot_root}models/{file}/best.pth")
            logging.info(f"{index + 1} {file} Training finish!... save training results")
        logging.info("Training finish!... save training results")
