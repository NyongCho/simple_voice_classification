from dataset import read_file, convert_mfcc, split_dataset
from model import k_fold_cross_validation

file_names = 'file_list.txt'
dir_path = 'dataset'

model_type = 'GMM'
sampling_rate = 16000
n_mfcc = 48
hop_length = 480
k_fold_split = 4

if __name__ == '__main__':
    audio_files = read_file(dir_path=dir_path, path=file_names)
    mfccs = convert_mfcc(audio_files=audio_files, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    splited_mfccs, targets = split_dataset(mfccs=mfccs, k=k_fold_split)
    accuracy = k_fold_cross_validation(splited_mfccs, targets, k_fold_split)
    print(f"Accuracy : {accuracy}")