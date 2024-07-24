import argparse
from reconstruction import Reconstruction
from upsampling import Upsampling


def get_parser():
    parser = argparse.ArgumentParser(description='RE-PU')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--k', type=int, default=16, help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--use_rotate', action='store_true', help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true', help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true', help='Jitter the pointcloud before training')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--downsampling', type=int, default=48)

    parser.add_argument('--feat_dims', type=int, default=512,  help='Number of dims for feature ')
    parser.add_argument('--upsampling_point', type=int, default=8192)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--snapshot_interval', type=int, default=1, help='Save snapshot interval ')

    parser.add_argument('--mode', type=str, default="multishape", choices=['multishape', 'oneshape'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--encoder', type=str, default='pct', choices=['pointnet', 'foldnet', 'dgcnn_cls', 'dgcnn_seg', 'oa', 'pct'])
    parser.add_argument('--prior', type=str, default='square_grid', 
                        choices=['square_grid', 'square_fib', 'square_ham', 'square_uniform', 'square_gaussian', 
                                 'disk_fib',
                                 'sphere_fib', 'sphere_uniform', 'sphere_gaussian'])
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the pointcloud before training')
    parser.add_argument('--progressive', action='store_true', help='Progressive training')
    parser.add_argument('--vq', action='store_true', help='Progressive training')
    parser.add_argument('--decoder', type=str, default='oa', choices=['mlp', 'foldnet', 'oa'])
    parser.add_argument('--loss', type=str, default='chamfer', choices=['chamfer', 'emd', 'l2'])
    parser.add_argument('--reconstruction_point', type=int, default=2048)
    parser.add_argument('--task', type=str, default='reconstruction', choices=['reconstruction', 'upsampling'])
    parser.add_argument('--pu1k_data', type=str, default="127_112_1936multishape.pt", help="PU1K data name")
    parser.add_argument('--epochs', type=int, default=10000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    if args.task == 'reconstruction':
        reconstruction = Reconstruction(args)
        if args.mode == "multishape":
            reconstruction.run_pu1k(args)
        else:
            reconstruction.run_oneshape(args)
    else:
        upsampling = Upsampling(args)
        if args.mode == "multishape":
            upsampling.run_pu1k(args)
        else:
            upsampling.run_oneshape(args)