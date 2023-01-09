import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from net import Auto_encoder, VAE
from trainer import Trainer
from dataset import ASDDataset
import utils

sep = os.sep

def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    # set device
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    csv_lines = []
    metrics = {'avg_auc': [], 'avg_pauc': []}
    result_path = os.path.join('./results', f'{args.version}', f'result_{args.pool_type}.csv')
    # load data
    # process_machine = ['ToyConveyor', 'fan', 'pump', 'slider']
    process_machine = ['valve']
    idx = 0
    reverse = False
    for (train_dir, add_dir, valid_dir, test_dir) in zip(
            sorted(args.train_dirs, reverse=reverse),
            sorted(args.add_dirs, reverse=reverse),
            sorted(args.valid_dirs, reverse=reverse),
            sorted(args.test_dirs, reverse=reverse)):
        args.logger.info(f'train and valid dirs: {train_dir}, {add_dir}, {valid_dir}')
        # args.logger.info(f'train and valid dirs: {train_dir}, {valid_dir}')
        args.machine = train_dir.split('/')[-2]
        # if idx >= len(process_machine) or args.machine != process_machine[idx]: continue
        # else: idx += 1
        # train_dataset = ASDDataset([train_dir], args)
        train_dataset = ASDDataset([train_dir, add_dir], args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
        # set model
        input_dims = (args.frames - 1) * args.n_mels if args.idnn else args.frames * args.n_mels
        output_dims = args.n_mels if args.idnn else input_dims
        net = Auto_encoder(input_dim=input_dims, output_dim=output_dims) if not args.vae else \
                VAE(input_dim=input_dims, output_dim=output_dims)
        if args.dp:
            net = nn.DataParallel(net, device_ids=args.device_ids)
        net = net.to(args.device)
        # optimizer & scheduler
        optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
        scheduler = None
        # trainer
        trainer = Trainer(args=args,
                          net=net,
                          optimizer=optimizer,
                          scheduler=scheduler)
        # train model
        if not args.load_epoch:
            trainer.train(train_dataloader, valid_dir)
        # load trained model
        load_epoch = args.load_epoch if args.load_epoch else 'best'
        model_path = os.path.join(args.writer.log_dir, 'model', args.machine, f'{load_epoch}_checkpoint.pth.tar')
        (trainer.net.module if args.dp else trainer.net).load_state_dict(torch.load(model_path)['model'])
        # valid model
        metric, csv_lines = trainer.valid(valid_dir=valid_dir, save=True, csv_lines=csv_lines)
        utils.save_csv(result_path, csv_lines, mode='w')
        for key in metrics.keys():
            metrics[key].append(metric[key])
        # test model
        trainer.test(test_dir)
    avg_auc, avg_pauc = np.mean(metrics['avg_auc']), np.mean(metrics['avg_pauc'])
    csv_lines.append(['Total Average', f'{avg_auc:.4f}', f'{avg_pauc:.4f}'])
    utils.save_csv(result_path, csv_lines, mode='w')


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    if args.idnn and args.vae: args.version = 'VIDNN'
    elif args.idnn: args.version = 'IDNN'
    elif args.vae: args.version = 'VAE'
    else: args.version = 'AE'
    args.version = f'{time_str}-{args.version}' if args.time_version else args.version
    args.version = args.version if not args.load_epoch else args.test_version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # save config file
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args)
    args.logger.info(args.version)
    main(args)


if __name__ == '__main__':
    run()