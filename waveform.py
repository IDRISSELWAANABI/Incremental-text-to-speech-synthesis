import torch
from model import vocoder
def generate_audio_pieces(spectrograms, delta_frames, vocoder, fixed_z_vector):

    extended_spectrograms = []
    num_words = len(spectrograms)
    
    for i in range(num_words):
        start_extension = spectrograms[i-1][-delta_frames:] if i > 0 else torch.Tensor([])
        end_extension = spectrograms[i+1][:delta_frames] if i < num_words-1 else torch.Tensor([])
        
        extended = torch.cat([start_extension, spectrograms[i], end_extension], dim=0)
        extended_spectrograms.append(extended)
    
    audio_pieces = [vocoder.generate_from_spectrogram(spec, fixed_z_vector) for spec in extended_spectrograms]
    final_audio = torch.cat(audio_pieces, dim=0)
    
    
    return final_audio


