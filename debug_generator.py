import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
import soundfile as sf
from utils.hparams import HParam

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(args, num, spk1, spk2, train):
    srate = args.sample_rate
    w1, _ = librosa.load(spk1, sr=srate)
    w2, _ = librosa.load(spk2, sr=srate)
    assert len(w2.shape) == len(w1.shape) == 1, \
        'wav files must be mono, not stereo'

    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    L = int(srate * args.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # save vad & normalized wav files
    dir_spk1 = os.path.join(args.out_dir, 's1')
    dir_spk2 = os.path.join(args.out_dir, 's2')
    dir_mix = os.path.join(args.out_dir, 'mix')

    spk1_wav_path = formatter(dir_spk1, '*-LibriSpeech.wav', num)
    spk2_wav_path = formatter(dir_spk2, '*-LibriSpeech.wav', num)
    mixed_wav_path = formatter(dir_mix, '*-LibriSpeech.wav', num)
    sf.write(spk1_wav_path, w1, srate)
    sf.write(spk2_wav_path, w2, srate)
    sf.write(mixed_wav_path, mixed, srate)


if __name__ == '__main__':
    # 解析命令行参数
    # 构造参数
    args = argparse.Namespace()
    # Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, dev-clean
    args.libri_dir = "/Users/mervin.qi/Desktop/PSE/Dataset/voicesplit_normalize"
    args.out_dir = "./gendata"
    #apply vad to wav file. yes(1) or no(0, default)
    args.vad = 0
    args.sample_rate = 16000
    args.audio_len = 3.0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'mix'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 's1'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 's2'), exist_ok=True)

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)]
                        # we recommned to exclude train-other-500
                        # See https://github.com/mindslab-ai/voicefilter/issues/5#issuecomment-497746793
                        # + \
                        #[x for x in glob.glob(os.path.join(args.libri_dir, 'train-other-500', '*'))
                        #    if os.path.isdir(x)]


    train_spk = [glob.glob(os.path.join(spk, '**', '*-norm.wav'), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    # 定义数据处理函数
    def train_wrapper(num, args, train_spk):
        s1, s2 = random.sample(train_spk, 2)
        spk1 = random.choice(s1)
        spk1 = random.choice(s2)
        mix(args, num,  spk1, spk1, train=True)

    # 遍历数据集并生成训练样本
    for i in tqdm.tqdm(range(10**1), desc='Generating train samples'):
        train_wrapper(i, args, train_spk)
