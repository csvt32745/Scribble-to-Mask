from argparse import ArgumentParser
import json

class HyperParameters():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()
        
        # Method / Network parameters
        parser.add_argument('--srb_3ch', help='split 3ch of sribble map', action='store_true')
        parser.add_argument('--backbone', help='backbone from timm', default='mobilenetv2_100')
        parser.add_argument('--pretrained_backbone', help='Use pretrained backbone from timm', default=False, action='store_true')
        
        # Data parameters
        parser.add_argument('--static_root', help='Static training data root', default='../static')
        parser.add_argument('--lvis_root', help='LVIS data root', default='../lvis')
        parser.add_argument('--matte_root', help='matte data root', default='../dataset_mat/VideoMatting108')

        # Generic learning parameters
        parser.add_argument('-i', '--iterations', help='Total number of iterations', default=80000, type=int)
        parser.add_argument('--lr', help='Learning rate', default=1e-4, type=float)
        parser.add_argument('--steps', help='Step at which the learning rate decays', nargs="*", default=[], type=int)
        parser.add_argument('--seg_epoch', help='Pretrain epoch for segmetation', default=5, type=int)

        parser.add_argument('-b', '--batch_size', help='Batch size', default=16, type=int)
        parser.add_argument('--gamma', help='Gamma used in learning rate decay', default=0.1, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_deeplab')
        parser.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # Multiprocessing parameters
        parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)

    def save(self, savepath):
        with open(savepath, 'w') as f:
            json.dump(self.args, f, indent=4) 