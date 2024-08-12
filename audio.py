import torchaudio
from torchaudio.transforms import MFCC


def transform_to_cv(audio):
    waveform, sample_rate = torchaudio.load(audio)
    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 512, "hop_length": 240,
                   "n_mels": 40, "norm": "slaney", 'mel_scale': 'slaney'},
    )
    torch_mfcc = mfcc_transform(waveform)
    return torch_mfcc

def transform_to_nlp(audio):
    file_name = audio.split("\\")[-1]
    print(file_name,' is readed!')
    file_path ='datasets/nlp/'+file_name
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return content
    except:
        return 'none'

    import os
    from os import path
    # from aip import AipSpeech
    # import time
    # APP_ID = '33812758  '
    # API_KEY = 'tlFzCYv1LV8LQuObcDxxbMEo'
    # SECRET_KEY = 'NXLqRZRXnw68jfd9yOVeiTUuvDrhpIaW'
    #
    # client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    # start_time = time.time()
    # ret = client.asr('datasets/trainset/class_one/V1.wav', 'pcm', 16000, {
    #     'dev_pid': 1537,
    # })
    # used_time = time.time() - start_time
    # print("used time: {}s".format(round(time.time() - start_time, 2)))
    # print('ret:{}'.format(ret))
    # return 'hello_world'

transform_to_cv('datasets/trainset/class_one/V1.wav')
transform_to_nlp('datasets/trainset/class_one/V1.wav')