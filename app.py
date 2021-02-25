from flask import Flask
from flask import request
from flask_cors import CORS
import librosa
import soundfile as sf
import wave
import torch
from werkzeug.datastructures import FileStorage
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import io

# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
	return "Hello World!"

@app.route('/transcribe', methods = ['POST'])

def postJsonHandler():
	vad = False
	speakerlist = ['Speaker']
	transcriptionlist = []
	file = request.files['file']

	with open("audio.wav", "wb") as aud:
		aud_stream = file.read()
		aud.write(aud_stream)
	audio_input, sr = librosa.load('audio.wav', sr=16000)
	
	input_values = tokenizer(audio_input, return_tensors="pt").input_values
	logits = model(input_values).logits
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = tokenizer.batch_decode(predicted_ids)[0]

	transcriptionlist.append(transcription)
	response = {'speakers': speakerlist, 'transcriptions': transcriptionlist}
	return response

if __name__ == '__main__':
	app.run()