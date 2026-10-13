import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')  # 5e-5
parser.add_argument('--batchsize', type=int, default=3, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load', type=str, default='/mnt/Data/HOME/gsx/MyNet/smt_tiny.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/mnt/Data/HOME/gsx/MyNet/datasets/train_dataset(new)/Imgs/', help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='/mnt/Data/HOME/gsx/MyNet/datasets/train_dataset(new)/GTs/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='/mnt/Data/HOME/gsx/MyNet/datasets/test_dataset/Imgs/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='/mnt/Data/HOME/gsx/MyNet/datasets/test_dataset/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='/mnt/Data/HOME/gsx/MyNet/pth(384)/', help='the path to save models and logs')
opt = parser.parse_args()