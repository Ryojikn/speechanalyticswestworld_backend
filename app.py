from flask import Flask
from flask import request
from flask_cors import CORS
import librosa
import soundfile as sf
import wave
import torch
from werkzeug.datastructures import FileStorage
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import io
from pprint import pprint

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')

(get_speech_ts,
 _, read_audio,
 _, _, _) = utils

# load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
	return "Hello World!"

@app.route('/transcribe', methods = ['POST'])

def postJsonHandler():
	vad = False
	transcriptionlist = []
	file = request.files['file']

	with open("audio.wav", "wb") as aud:
		aud_stream = file.read()
		aud.write(aud_stream)

	audio_input, sr = librosa.load('audio.wav', sr=16000)
	segments = get_speech_ts(read_audio('audio.wav'), model=vad_model, num_steps=1)
	audio_segments = [audio_input[segment['start']:segment['end']] for segment in segments]
	
	for audio_seg in audio_segments:
		input_values = tokenizer(audio_seg, return_tensors="pt").input_values
		logits = model(input_values).logits
		predicted_ids = torch.argmax(logits, dim=-1)
		transcription = tokenizer.batch_decode(predicted_ids)[0]
		transcriptionlist.append(transcription)

	response = {'speakers': ['Speaker']*len(transcriptionlist), 'transcriptions': transcriptionlist}
	return response

if __name__ == '__main__':
	app.run()