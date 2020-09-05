from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
import yaml 
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', 
            default='/work/b07u1234/b06502162/HW2-1/github-HW2-1/vctk/trimmed_vctk_spectrograms/sr_24000_mel_norm')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-train_index_file', default='train_samples_128.json')
    parser.add_argument('-logdir', default='log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('-store_model_path', default='/work/b07u1234/b07505026/hw2-2/model/origin')
    parser.add_argument('-load_model_path', default='/work/b07u1234/b07505026/hw2-2/model/origin')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=10000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=200000, type=int)

    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    solver = Solver(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)
