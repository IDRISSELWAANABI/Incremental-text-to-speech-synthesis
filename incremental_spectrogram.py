import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixToPrefixTTS(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(PrefixToPrefixTTS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.spectrogram_frame_predictor = nn.Linear(hidden_size, spectrogram_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.stop_predictor = nn.Linear(hidden_size, 1)
        
    def forward(self, x, max_frames):
        embeddings = self.embedding(x)
        encoder_outputs, _ = self.encoder(embeddings)
        
        decoder_input = torch.zeros((x.size(0), 1, encoder_outputs.size(2)), device=x.device)
        hidden = None
        spectrograms = []
        attention_weights = []
        
        for _ in range(max_frames):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            attention_scores = self.attention(torch.cat((decoder_output.repeat(1, encoder_outputs.size(1), 1), encoder_outputs), -1))
            attention_scores = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_scores * encoder_outputs, dim=1, keepdim=True)
            
            spectrogram_frame = self.spectrogram_frame_predictor(context_vector)
            stop_probability = torch.sigmoid(self.stop_predictor(context_vector))
            
            spectrograms.append(spectrogram_frame)
            attention_weights.append(attention_scores)

            if stop_probability > 0.5:
                break

            decoder_input = context_vector

        return torch.cat(spectrograms, dim=1), torch.cat(attention_weights, dim=1)

vocab_size = 100  
embed_size = 256
hidden_size = 512
num_layers = 2
spectrogram_size = 80  
max_frames = 100  

model = PrefixToPrefixTTS(vocab_size, embed_size, hidden_size, num_layers)
text_input = torch.randint(0, vocab_size, (1, 10))  
spectrograms, attention_weights = model(text_input, max_frames)
