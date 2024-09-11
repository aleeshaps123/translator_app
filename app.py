import streamlit as st
import torch
import torch.nn as nn
from googletrans import Translator,LANGUAGES

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        
        return outputs
import streamlit as st
import torch
import torch.nn as nn
from googletrans import Translator

# Define the Encoder, Decoder, and Seq2Seq classes as in your code

# Apply custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .stTextInput>div>input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s, box-shadow 0.3s;
    }

    .stTextInput>div>input:focus {
        border-color: #388E3C;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        outline: none;
    }

    .stSelectbox>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s, box-shadow 0.3s;
    }

    .stSelectbox>div>div>input:focus {
        border-color: #388E3C;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        outline: none;
    }

    .stTitle {
        color: #4CAF50;
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
    }

    .stWrite {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit Interface
st.title("Language Translator")
st.write("Enter the text to translate and select the target language:")

translator = Translator()

input_text = st.text_input("Input Text")

languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Chinese (Simplified)': 'zh-cn',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Italian': 'it',
    'Malayalam': 'ml',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn'
}

target_language = st.selectbox("Select Target Language", list(languages.keys()))

if st.button("Translate"):
    if input_text:
        try:
            translation = translator.translate(input_text, dest=languages[target_language])
            st.write(f"Translated Text: {translation.text}")
        except Exception as e:
            st.write(f"An error occurred: {e}")
