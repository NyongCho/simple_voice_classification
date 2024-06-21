import numpy as np
import librosa

# class VoiceDataset(Dataset):

def read_file(dir_path, path):
    file_path = path
    audio_files = []

    with open(file_path, 'r') as file:
        for line in file:
            audio_files.append(dir_path+'/'+line.strip())

    return audio_files

def convert_mfcc(audio_files, sr, n_mfcc, hop_length):
    mfccs = []
    n_mfcc = n_mfcc
    hop_length = hop_length

    for file in audio_files:
        y, sr = librosa.load(file, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
        mfccs.append(mfcc)
        
    return mfccs

def split_dataset(mfccs, k):
    length = len(mfccs[0])//k

    for i, mfcc in enumerate(mfccs):
        tmp = []
        for j in range(k):
            tmp.append(np.array(mfcc[length*(j):length*(j+1)]))
        mfccs[i] = np.array(tmp)
    mfccs = np.array(mfccs)

    targets = np.array([np.array([np.full((length,1), i) for _ in range(k)]) for i in range(10)])

    return [mfccs, targets]