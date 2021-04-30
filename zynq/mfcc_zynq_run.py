import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc


wav_path = '/share/Downloads/speech/yes/ffd2ba2f_nohash_0.wav'
# wav_path = '/share/Downloads/speech/go/d9d08edd_nohash_0.wav'
# wav_path = '0a2b400e_nohash_0.wav' 

def psf_mfcc(wav_path):
    frequency_sampling, x = wavfile.read(wav_path)
    # print(frequency_sampling)
    # print(x.shape, x.dtype)

    x = np.pad(x, (0, 16000 - x.shape[0]), mode='constant')
    # print(x.shape, x.dtype)
    
    if x.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    audio_signal = x / max_nb_bit # samples is a numpy array of float representing the samples
    clamp_audio = audio_signal.clip(-1., 1.)

    features_mfcc = mfcc(clamp_audio, samplerate=16000, 
                        winlen=0.04, winstep=0.02, numcep=10, nfilt=40, lowfreq=20, highfreq=4000,
                        nfft=800, appendEnergy=False, preemph=0, ceplifter=0)
    # print(features_mfcc.shape)

    # outputs = features_mfcc.flatten()
    # print(outputs.shape)
    # print(outputs)

    return features_mfcc, audio_signal

def tf_mfcc(wav_filename_placeholder_, foreground_volume_placeholder_, time_shift_padding_placeholder_, time_shift_offset_placeholder_):
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=16000)

    # Allow the audio sample's volume to be adjusted.
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    foreground_volume_placeholder_)

    # Shift the sample's start position, and pad any gaps with zeros.
    padded_foreground = tf.pad(
        scaled_foreground,
        time_shift_padding_placeholder_,
        mode='CONSTANT')
    # tf.slice 从inputs中抽取部分内容
    sliced_foreground = tf.slice(padded_foreground,
                                 time_shift_offset_placeholder_,
                                 [16000, -1])
    background_clamp = tf.clip_by_value(sliced_foreground, -1.0, 1.0)

    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=640,
        stride=320,
        magnitude_squared=True)
    mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=10)
    
    return mfcc_, wav_decoder.audio, scaled_foreground

features1, wav_data1 = psf_mfcc(wav_path)

print(features1.shape, wav_data1.shape)