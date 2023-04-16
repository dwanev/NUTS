import os
import wave
import math
import torch
import glob
import seaborn
import logging
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Union

N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

def _delete_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of
    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported
    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """

    if not torch.is_tensor(audio):
        raise Exception('Need a torch tensor')

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    ls_max = log_spec.max()
    log_spec = torch.maximum(log_spec, ls_max.clone().detach() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def convert_mono_16k_wav(in_filename, out_filename):
    # cmd line example:
    # ffmpeg -i /home/stduser/recordings/Tues_17_49_local_True.wav  -b:a 192k ./recordings/Tues_17_49_local_True.mp3
    _delete_file(out_filename)
    logging.info('Converting : ' + in_filename + ' to 16KHz mono wav: ' + out_filename)
    p3 = subprocess.Popen(
        ['ffmpeg', '-i', in_filename, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', out_filename],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p3.wait()  # wait for the process to complete before returning


def load_file_return_torch_tensor(filename):
    chunk_size = 32000
    audio_stream = wave.open(
        filename,
        'rb')
    data = audio_stream.readframes(chunk_size)
    # pickle_encoded_data = pickle.dumps(data)
    np_audio_data = np.frombuffer(data, dtype=np.int16)
    np_audio_data_float_normalised = np.frombuffer(np_audio_data, np.int16).flatten().astype(np.float32) / 32768.0
    torch_data = torch.from_numpy(np_audio_data_float_normalised)
    return torch_data


def normalise_torch(a):
    a -= torch.min(a)
    a /= torch.max(a)
    return a


def to_str(f):
    return "{:.2f}".format(f)

def filename_to_instance_name(fn):
    """
    Get the filename (remove the path), remove all problematic characters
    """
    utterance_label = fn[fn.rfind('/') + 1:]
    instance_name = utterance_label.replace('_', '').replace('.wav', '')[:32]
    return instance_name

def value_to_narsese(value, index, instance_name, prop_prefix="prop", tense = "", truth_threshold=0.0):

    if tense != "":
        tense = " "+tense

    property_name = prop_prefix + str(index)
    t = value.item()
    if math.isnan(t):
        t = 0.0
    if t < 0.5:
        t = 1.0 - t
        property_name = "NOT"+property_name
    if t >= truth_threshold:
        statement = '<{' + instance_name + '} --> [' + property_name + ']>.'+tense+' %' + to_str(t) + '%'

    return statement, instance_name


def create_narsese_isa(instance_name, is_a_what, tense = "", tv=0.9):
    if tense != "":
        tense = " "+tense
    return '<{' + instance_name + '} --> ' + is_a_what + '>.'+tense+' %' + to_str(tv) + '%'


def create_narsese_isa_question(instance_name, is_a_what, tense = ""):
    if tense != "":
        tense = " "+tense
    return '<{' + instance_name + '} --> ' + is_a_what + '>?'+tense


def create_narsese_whatis_question(instance_name, tense = ""):
    if tense != "":
        tense = " "+tense
    return '<{' + instance_name + '} --> ?1>?'+tense


def create_narsese_islike_statement(instance_name1, instance_name2, tense = "", tv=0.9):
    if tense != "":
        tense = " "+tense
    statement = '<{' + instance_name1 + '} <-> {'+instance_name2+'}>.'+tense
    if tv is not None:
        statement += ' %' + to_str(tv) + '%'
    return statement

def create_narsese_islike_question(instance_name, tense = ""):
    if tense != "":
        tense = " "+tense
    return '<{' + instance_name + '} <-> {?1}>?'+tense


def create_narsese_islike_question2(instance_name1, instance_name2, tense = ""):
    if tense != "":
        tense = " "+tense
    return '<{' + instance_name1 + '} <-> {' + instance_name2 + '}>?'+tense



def mel_to_narsese(mel, index, utterance_label, normalised_x, mel_threshold=0.5, label='?', statements_by_property_name = {}, tense = "", truth_threshold=0.0):
    instance_name = utterance_label.replace('_','').replace('.wav','')[:32]
    if tense != "":
        tense = " "+tense
    for r in range(80):
        property_name = 'mel' + str(r)+'x' + str(normalised_x)
        t = mel[r, index].item()
        if t < 0.5:
            t = 1.0 - t
            property_name = "NOT"+property_name
        if t >= truth_threshold:
            statement = '<{' + instance_name + '} --> [' + property_name + ']>.'+tense+' %' + to_str(t) + '%'
            if property_name not in statements_by_property_name:
                statements_by_property_name[property_name] = statement  # only take one value if more than 1 are calculated and collide

            # if t >= 0.9: # if it is a strong signal, connect to label. TODO tune this threshold?
            #     statement = '<[' + property_name + '] --> '+label+'>.'+tense+' %' + to_str(t*0.9) + '%'  # this of cause is not 100%
            #     property_name2 = property_name +"2"+label
            #     if property_name2 not in statements_by_property_name:
            #         statements_by_property_name[property_name2] = statement  # only take one value if more than 1 are calculated and collide

    return statements_by_property_name, instance_name


def convert_file_to_inputs(fn, mel, out_filename, break_at_column=-1, label="NO_LABEL", truth_threshold=0.0, max_frames_per_second=10):
    """
    Build and return the NARs from the MEL spectrum.

    """
    mel = normalise_torch(mel) # between 0 and 1.0

    # print(mel.shape)
    # print(mel)
    # TODO need to workout a way to do this.
    # TODO represent 80 float -1 - +1.0 as text (of truth?)

    instance_name_list = []

    statements_by_property_name = {}
    all_statement_list = []
    for c in range(mel.shape[1]):
        normalised_x = int(max_frames_per_second * c/mel.shape[1])
        statements_by_property_name, instance_name = mel_to_narsese(mel, index=c, utterance_label=fn[fn.rfind('/')+1:], normalised_x=normalised_x, statements_by_property_name=statements_by_property_name, label=label, truth_threshold=truth_threshold)

        if break_at_column > 0 and break_at_column == c:
            break

    instance_name_list.append(instance_name)

    all_statement_list.extend(statements_by_property_name.values())
    # label the instance via a IS A statement
    if label is not None and label != "NO_LABEL":
        statement = create_narsese_isa(instance_name, is_a_what=label, tense = "", tv=0.9)
        all_statement_list.append(statement)

    with open(out_filename, 'a') as f:
        for statement in all_statement_list:
            f.write(statement + '\n')

    # print("Number of statements", len(statements_by_property_name))
    # print(instance_name_list)
    return instance_name_list, statements_by_property_name.values(), all_statement_list, len(statements_by_property_name)


def convert_tensor_to_statements(instance_name, reduced_dim_norm, out_filename=None, truth_threshold=0.0, add_isa_statement=False, label = "fred"):
    """
    Build and return the normalized the MEL spectrum.

    """
    instance_name_list = []
    statement_list = []
    for c in range(reduced_dim_norm.shape[0]):
        statement, instance_name = value_to_narsese(reduced_dim_norm[c], index=c,
                                                    instance_name=instance_name,
                                                    truth_threshold=truth_threshold,
                                                    prop_prefix="prop")

        instance_name_list.append(instance_name)
        statement_list.append(statement)
    if add_isa_statement:
        statement = create_narsese_isa(instance_name, is_a_what=label, tense = "", tv=0.9)
        statement_list.append(statement)

    if out_filename is not None:
        with open(out_filename, 'a') as f:
            for statement in statement_list:
                f.write(statement + '\n')


    # print("Number of statements", len(statement_list))
    # print(instance_name_list)
    return instance_name_list, statement_list



def get_stats(mel):
    """
    Build and return the NARs from the MEL spectrum.

    """
    mel = normalise_torch(mel) # between 0 and 1.0


    return {
        'average':mel.numpy().mean(),
        'num_gt_mean': (mel.numpy() > mel.numpy().mean()).sum()
    }



if __name__ == '__main__':
    out_filename = 'out.nal'
    _delete_file(out_filename)
    all_instance_names = []
    break_at_column = -1
    truth_threshold = 0.9
    target_word = 'one'
    mels = {}
    stats = {}
    for i, fn in enumerate(glob.glob("/home/dwane/Downloads/speech_commands_v0.02/"+target_word+"/*.wav")):
        torch_data = load_file_return_torch_tensor(fn)
        mel = log_mel_spectrogram(torch_data, n_mels=N_MELS)
        stats[i] = get_stats(mel)
        mels[i] = mel.numpy().reshape(-1)
        instance_name_list, _ = convert_file_to_inputs(fn, mel, out_filename, break_at_column=break_at_column, label=target_word, truth_threshold=truth_threshold)
        all_instance_names.extend(instance_name_list)
        if i >= 200:
            break

    # correlate
    # np.corrcoef(mels[0].numpy().reshape(-1), mels[3].numpy().reshape(-1))
    corr = np.zeros( (i,i) )
    for j in range(i):
        for k in range(i):
            if k < j:
                mel_len = 8000
                if len(mels[k]) != mel_len:
                    mels[k] = np.pad(mels[k], (0, mel_len-len(mels[k])), 'constant', constant_values=(0, 0))
                if len(mels[j]) !=mel_len:
                    mels[j] = np.pad(mels[j], (0, mel_len-len(mels[j])), 'constant', constant_values=(0, 0))

                corr[k,j] = np.corrcoef(mels[k], mels[j])[0,1]

    plt.clf()
    svm = seaborn.heatmap(data=corr )
    plt.savefig('experiment_2_correlation_heatmap_'+target_word+'.png')

    print(corr)

    with open(out_filename, 'a') as f:
        for in1 in all_instance_names:
            statement = '<{'+in1+'} <-> ?1>?'
            f.write(statement + '\n')

        for in1 in all_instance_names:
            for in2 in all_instance_names:
                if in1 < in2:
                    statement = '//<{'+in1+'} <-> {'+in2+'}>?'
                    f.write(statement + '\n')


