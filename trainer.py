import os
import glob
import librosa
import sklearn
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import joblib

from loss import ASDLoss
import utils


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(self.args.device)
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr, power=self.args.power,
                                      n_fft=self.args.n_fft, n_mels=self.args.n_mels,
                                      win_length=self.args.win_length, hop_length=self.args.hop_length)
    def train(self, train_loader, valid_dir):
        # self.valid(valid_dir, save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model', self.args.machine)
        os.makedirs(model_dir, exist_ok=True)
        n_mels = self.args.n_mels
        frames = self.args.frames
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'{self.args.machine}|Epoch-{epoch}')
            for (x_mels) in train_bar:
                # forward
                b, n, d = x_mels.shape
                inputs = x_mels.reshape(b*n, d).float()
                targets = inputs
                if self.args.idnn:
                    cf_mask = torch.ones_like(x_mels)
                    cf_mask[:, :, n_mels * (frames // 2): n_mels * (frames // 2 + 1)] = 0
                    inputs = x_mels[cf_mask.bool()].reshape(b*n, -1)
                    targets = x_mels[~cf_mask.bool()].reshape(b*n, -1)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                if not self.args.vae:
                    outputs, _ = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                else:
                    outputs, _, mu, logvar = self.net(inputs)
                    loss = self.criterion(outputs, targets, mu=mu, logvar=logvar)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'{self.args.machine}/train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                metric, _ = self.valid(valid_dir, save=False)
                avg_auc, avg_pauc = metric['avg_auc'], metric['avg_pauc']
                self.writer.add_scalar(f'{self.args.machine}/auc_s', avg_auc, epoch)
                self.writer.add_scalar(f'{self.args.machine}/pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)

    def valid(self, valid_dir, save=True, result_dir=None, csv_lines=[]):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        n_mels = self.args.n_mels
        frames = self.args.frames
        metric = {}
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        target_dir = valid_dir
        start = time.perf_counter()
        machine_type = target_dir.split('/')[-2]
        machine_id_list = utils.get_machine_id_list(target_dir)
        # print(machine_id_list)
        csv_lines.append([machine_type])
        csv_lines.append(['ID', 'AUC', 'pAUC'])
        performance = []
        for id_str in machine_id_list:
            csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
            test_files, y_true = utils.get_valid_file_list(target_dir, id_str)
            y_pred = [0. for _ in test_files]
            anomaly_score_list = []
            for file_idx, file_path in enumerate(test_files):
                x_mels = self.transform(file_path)
                n, d = x_mels.shape
                inputs = x_mels.float()
                targets = inputs
                if self.args.idnn:
                    cf_mask = torch.ones_like(x_mels)
                    cf_mask[:, n_mels * (frames // 2): n_mels * (frames // 2 + 1)] = 0
                    inputs = x_mels[cf_mask.bool()].reshape(n, -1)
                    targets = x_mels[~cf_mask.bool()].reshape(n, -1)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                with torch.no_grad():
                    outputs, _ = net(inputs)
                y_pred[file_idx] = utils.calculate_anomaly_score(outputs.cpu().numpy(), targets.cpu().numpy(),
                                                                 frames=1 if self.args.idnn else frames, n_mels=n_mels,
                                                                 pool_type=self.args.pool_type, decay=self.args.decay)
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            if save: utils.save_csv(csv_path, anomaly_score_list)
            # compute auc and pAuc
            auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
            p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
            performance.append([auc, p_auc])
            csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
        # calculate averages for AUCs and pAUCs
        # print(performance)
        amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
        mean_auc, mean_p_auc = amean_performance[0], amean_performance[1]
        # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
        time_nedded = time.perf_counter() - start
        csv_lines.append(["Average"] + list(amean_performance))
        csv_lines.append([])
        self.logger.info(f'Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc: {mean_auc:.3f}\tavg_pauc: {mean_p_auc:.3f}')
        print(f'Test time: {time_nedded:.2f} sec')
        metric['avg_auc'], metric['avg_pauc'] = mean_auc, mean_p_auc
        return metric, csv_lines

    def test(self, test_dir, result_dir=None):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        metric = {}
        n_mels = self.args.n_mels
        frames = self.args.frames
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./evaluator/teams', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        target_dir = test_dir
        machine_type = target_dir.split('/')[-2]
        machine_id_list = utils.get_machine_id_list(target_dir)
        for id_str in machine_id_list:
            csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
            test_files = utils.get_test_file_list(target_dir, id_str)
            y_pred = [0. for _ in test_files]
            anomaly_score_list = []
            for file_idx, file_path in enumerate(test_files):
                x_mels = self.transform(file_path)
                n, d = x_mels.shape
                inputs = x_mels.float()
                targets = inputs
                if self.args.idnn:
                    cf_mask = torch.ones_like(x_mels)
                    cf_mask[:, n_mels * (frames // 2): n_mels * (frames // 2 + 1)] = 0
                    inputs = x_mels[cf_mask.bool()].reshape(n, -1)
                    targets = x_mels[~cf_mask.bool()].reshape(n, -1)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                with torch.no_grad():
                    outputs, _ = net(inputs)
                y_pred[file_idx] = utils.calculate_anomaly_score(outputs.cpu().numpy(), targets.cpu().numpy(),
                                                                 frames=1 if self.args.idnn else frames, n_mels=n_mels,
                                                                 pool_type=self.args.pool_type, decay=self.args.decay)
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            utils.save_csv(csv_path, anomaly_score_list)


    def transform(self, filename):
        (x, _) = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        dims = self.args.n_mels * self.args.frames
        vector_size = x_mel.shape[1] - self.args.frames + 1
        mel_vec = torch.zeros((vector_size, dims))
        for t in range(self.args.frames):
            mel_vec[:, t * self.args.n_mels: (t + 1) * self.args.n_mels] = x_mel[:, t: t + vector_size].T
        return mel_vec
