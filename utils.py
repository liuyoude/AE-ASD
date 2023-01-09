"""
functional functions
"""
import itertools
import os
import re
import shutil
import glob
import yaml
import csv
import logging
import random
import numpy as np
import torch
import torchaudio
from scipy.special import psi, polygamma

sep = os.sep


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_load_version_files(path, file_patterns, pass_dirs=None):
    #    save latest version files
    if pass_dirs is None:
        pass_dirs = ['.', '_', 'runs', 'results']
    copy_files(f'.{sep}', 'runs/latest_project', file_patterns, pass_dirs)
    copy_files(f'.{sep}', os.path.join(path, 'project'), file_patterns, pass_dirs)


def save_csv(file_path, data: list, mode='w'):
    with open(file_path, mode, newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


# 复制目标文件到目标路径
def copy_files(root_dir, target_dir, file_patterns, pass_dirs=['.git']):
    # print(root_dir, root_dir.split(sep), [name for name in root_dir.split(sep) if name != ''])
    os.makedirs(target_dir, exist_ok=True)
    len_root = len([name for name in root_dir.split(sep) if name != ''])
    for root, _, _ in os.walk(root_dir):
        cur_dir = sep.join(root.split(sep)[len_root:])
        first_dir_name = cur_dir.split(sep)[0]
        if first_dir_name != '':
            if (first_dir_name in pass_dirs) or (first_dir_name[0] in pass_dirs): continue
        # print(len_root, root, cur_dir)
        target_path = os.path.join(target_dir, cur_dir)
        os.makedirs(target_path, exist_ok=True)
        files = []
        for file_pattern in file_patterns:
            file_path_pattern = os.path.join(root, file_pattern)
            files += sorted(glob.glob(file_path_pattern))
        for file in files:
            target_path_file = os.path.join(target_path, os.path.split(file)[-1])
            shutil.copyfile(file, target_path_file)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list


def set_type(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_machine_id_list(target_dir,
                        dir_name='test',
                        ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path])
    )))
    return machine_id_list


def get_valid_file_list(target_dir,
                        id_name,
                        dir_name='test',
                        prefix_normal='normal',
                        prefix_anomaly='anomaly',
                        ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


def get_test_file_list(target_dir,
                       id_name,
                       dir_name='test',
                       ext='wav'):
    files_path = f'{target_dir}/{id_name}*.{ext}'
    files = sorted(glob.glob(files_path))
    return files


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))





def calculate_anomaly_score(data,
                            predict_data,
                            frames=5,
                            n_mels=128,
                            pool_type='mean',
                            decay=0.99):
    bs = data.shape[0]
    data = data.reshape(bs, frames, n_mels)
    predict_data = predict_data.reshape(bs, frames, n_mels)
    # mean score of n_mels for every frame
    errors = np.mean(np.square(data - predict_data), axis=2).reshape(bs, frames)
    # mean score of frames
    errors = np.mean(errors, axis=1)
    #errors = np.max(errors, axis=1)

    if pool_type == 'mean':
        score = np.mean(errors)
    elif pool_type == 'max':
        score = np.max(errors)
    elif pool_type == 'gwrp':
        score = calculate_gwrp(errors, decay)
    else:
        raise Exception(f'the pooling type is {pool_type}, mismatch with mean, max, max_frames_mean, gwrp, and gt_mean')

    return score



# move data for dataset
def move_dataset():
    eval_ids_dict = {
        'fan': ['01', '03', '05'],
        'pump': ['01', '03', '05'],
        'slider': ['01', '03', '05'],
        'valve': ['01', '03', '05'],
        'ToyCar': ['05', '06', '07'],
        'ToyConveyor': ['04', '05', '06'],
    }
    source_dir = '../../data/dataset'
    target_dir = '../../data/eval_dataset'
    for machine_type in eval_ids_dict.keys():
        s_m_dir = os.path.join(source_dir, machine_type, 'train')
        t_m_dir = os.path.join(target_dir, machine_type, 'train')
        os.makedirs(t_m_dir, exist_ok=True)
        for id in eval_ids_dict[machine_type]:
            file_pattern = os.path.join(s_m_dir, f'*id_{id}*.wav')
            files = glob.glob(file_pattern)
            for file in files:
                _, fname = os.path.split(file)
                target_path = os.path.join(t_m_dir, fname)
                shutil.move(file, target_path)


if __name__ == '__main__':
    # print(get_filename_list('../Fastorch', ext='py'))
    move_dataset()
