import torch
import numpy as np
from tacotron2 import Tacotron2
from tacotron2.text import text_to_sequence
from parallel_wavegan.utils import load_model
import soundfile as sf

tacotron2 = Tacotron2()
tacotron2.load_state_dict(torch.load('tacotron2_statedict.pt'))
tacotron2.eval()

def synthesize_text_to_mel(text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).long()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)

    return mel_outputs_postnet

vocoder = load_model("checkpoint-400000steps.pkl").to("cuda").eval()
vocoder.remove_weight_norm()

def mel_to_waveform(mel_spectrogram):
    with torch.no_grad():
        c = mel_spectrogram.to("cuda")
        y_hat = vocoder.inference(c)
        y_hat = y_hat.view(-1).cpu().numpy()

    return y_hat

text = "Hello, world!"
mel_spectrogram = synthesize_text_to_mel(text)
waveform = mel_to_waveform(mel_spectrogram)

sf.write('output.wav', waveform, 22050)
