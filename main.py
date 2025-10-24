import os
import streamlit as st
import soundfile as sf
import tempfile
import numpy as np
import time
import torch
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Realtime Emotion Detection", layout="wide")
st.title("🎤 Speech Emotion Detection")

# ----------------------------
# Parameters
# ----------------------------
SAMPLE_RATE = 16000
DURATION = 5.0
SEQ_LEN = 16
FEATURE_DIM = 48
POOL_AUDIO = True

fusion_model_dir = "final"
fusion_model_config_path = os.path.join(fusion_model_dir, "fusion_model_config.json")
fusion_model_weights_path = os.path.join(fusion_model_dir, "fusion_model.weights.h5")
audio_scaler_path = os.path.join(fusion_model_dir, "audio_scaler.pkl")
text_scaler_path = os.path.join(fusion_model_dir, "text_scaler.pkl")

# ----------------------------
# Load models (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    print("Loading models")
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
    from sentence_transformers import SentenceTransformer
    from tensorflow.keras.models import model_from_json
    import pickle

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ASR model
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(DEVICE)
    asr_model.eval()

    # Audio embedding model
    embed_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    embed_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    embed_model.eval()

    # Sentence BERT
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # Fusion model
    with open(fusion_model_config_path, "r") as f:
        model_json = f.read()
    fusion_model = model_from_json(model_json)
    fusion_model.load_weights(fusion_model_weights_path)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Scalers
    with open(audio_scaler_path, 'rb') as f:
        audio_scaler = pickle.load(f)
    with open(text_scaler_path, 'rb') as f:
        text_scaler = pickle.load(f)

    return asr_processor, asr_model, embed_processor, embed_model, sbert, fusion_model, audio_scaler, text_scaler, DEVICE

asr_processor, asr_model, embed_processor, embed_model, sbert, fusion_model, audio_scaler, text_scaler, DEVICE = load_models()



# ----------------------------
# Inference function
# ----------------------------
def infer_once(wav_path):
    import librosa

    # 1. Transcribe
    speech, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    input_values = asr_processor(speech, sampling_rate=sr, return_tensors="pt", padding="longest").input_values.to(DEVICE)
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0].lower().strip()

    # 2. Audio embedding
    input_values_embed = embed_processor(speech, sampling_rate=sr, return_tensors="pt", padding="longest").input_values.to(DEVICE)
    with torch.no_grad():
        last_hidden = embed_model(input_values_embed).last_hidden_state
        audio_emb = last_hidden.mean(dim=1).squeeze().cpu().numpy()
    audio_emb_seq = audio_scaler.transform(audio_emb.reshape(1, -1)).reshape(1, SEQ_LEN, FEATURE_DIM)

    # 3. Text embedding
    if len(transcription.strip()) == 0:
        text_emb = np.zeros((sbert.get_sentence_embedding_dimension(),))
    else:
        text_emb = sbert.encode([transcription], convert_to_numpy=True, show_progress_bar=False)[0]
    text_emb_seq = text_scaler.transform(text_emb.reshape(1, -1))

    # 4. Predict
    pred_prob = fusion_model.predict([audio_emb_seq, text_emb_seq], verbose=0)[0]
    pred_idx = np.argmax(pred_prob)
    label_classes = ['anger', 'joy', 'neutral', 'sadness']
    pred_label = label_classes[pred_idx]

    return transcription, pred_label, pred_prob[pred_idx]

# ----------------------------
# Streamlit UI
# ----------------------------
# Upload or Record
st.info("Provide your audio input")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.write(uploaded_file.getbuffer())
    wav_path = tmp_file.name
    st.audio(wav_path, format="audio/wav")

else:
    wav_path = None

if wav_path is not None and st.button("Predict Emotion"):
    transcription, emotion, prob = infer_once(wav_path)
    st.write(f"**Transcription:** {transcription}")
    st.write(f"**Predicted emotion:** {emotion}")
    st.write(f"**Probability:** {prob:.3f}")
    st.warning("⚠️ Predicted emotion depends on voice tone, loudness, and background noise.")
