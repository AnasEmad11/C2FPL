import argparse

parser = argparse.ArgumentParser(description='RFS_AD')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')

parser.add_argument('--rgb-list', default='./list/ucf-c3d.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='./list/ucf-c3d-test.list', help='list of test rgb features ')
parser.add_argument('--gt', default='list/gt-ucf_RTFM.npy', help='file of ground truth ')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.0001)')

parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=0, help='number of workers in dataloader')
parser.add_argument('--model-name', default='C2FPL', help='name to save model')

parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--datasetname', default='UCF', help='dataset to train on (default: )')

parser.add_argument('--max-epoch', type=int, default=30, help='maximum iteration to train (default: 100)')

parser.add_argument('--optimizer', default='SGD', help='Number of segments of each video')
parser.add_argument('--lossfn', default='BCE', help='Number of segments of each video')
parser.add_argument('--stepsize',type=int,  default=5, help='lr_scheduler stepsize')
parser.add_argument('--pseudo',type=str,  default='Unsup_labels/UCF_unsup_labels_original_V2.npy', help='pseudo labels')


parser.add_argument('--windowsize',type=float,  default=0.09, help='lr_scheduler stepsize')
parser.add_argument('--modelversion',type=str,  default='Model_V2', help='Model version')
parser.add_argument('--eps2',type=float,  default=0.4, help='lr_scheduler stepsize')
parser.add_argument('--outer-epochs',type=int,  default=1, help='lr_scheduler stepsize')
parser.add_argument('--pseudofile',type=str,  default='UCF_unsup_labels_original_V2', help='ground truth file')
parser.add_argument('--conall',type=str,  default='concat_UCF', help='ground truth file')

