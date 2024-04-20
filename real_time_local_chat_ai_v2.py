import time
import numpy as np
import pyaudio
import onnxruntime as rt
import pvporcupine
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import multiprocessing as mp
import time
import pyttsx3
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import wave
import os
import requests

# source_dir = os.getcwd()

# VAD
class VADDetector2:

    def __init__(self, model_path, chunk_size=512, format=pyaudio.paFloat32, channels=1, rate=16000):
        self.model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=self._get_session_options())
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.h_state = np.zeros((2, 1, 64)).astype('float32')
        self.c_state = np.zeros((2, 1, 64)).astype('float32')

    def _get_session_options(self):
        opts = rt.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        return opts

    def process_audio_chunk(self, audio_chunk):
        ort_inputs = {
            'input': audio_chunk.reshape(1, -1),
            'h': self.h_state,
            'c': self.c_state,
            'sr': np.array(self.rate, dtype='int64')
        }
        ort_outs = self.model.run(None, ort_inputs)
        self.h_state, self.c_state = ort_outs[1], ort_outs[2]
        return ort_outs[0]

    def detect_voice_activity(self, data):

        try:

            audio_chunk = np.frombuffer(data, dtype=np.float32)
            output = self.process_audio_chunk(audio_chunk)
                      
            if output.item() < 0.5:

                vad_sentance_end_bool = True

            else:
                vad_sentance_end_bool = False

        except KeyboardInterrupt:
            print("Exiting program...")

        return vad_sentance_end_bool

# WAKE WORD
def detect_wake_word(jarvis_handle):

    global wake_word_bool

    # Define a function to get the next audio frame
    def get_next_audio_frame(stream, chunk_size):
        data = stream.read(chunk_size)
        return np.frombuffer(data, dtype=np.int16)

    # Set up PyAudio stream
    p = pyaudio.PyAudio()
    chunk_size = 512
    format = pyaudio.paInt16
    channels = 1
    rate = 16000

    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    try:
        print("Listening for wake word...")
        while True:
            
            audio_frame = get_next_audio_frame(stream, chunk_size)
            keyword_index = jarvis_handle.process(audio_frame)

            if keyword_index >= 0:
                wake_word_bool = True
                break

            else:
                wake_word_bool = False
              
    except KeyboardInterrupt:
        print("Interrupted by user")
        exit()

    finally:

        stream.stop_stream()
        stream.close()
        p.terminate()


def run_vad(vad_detector, audio_chunk):
    return vad_detector.detect_voice_activity(audio_chunk)


# STT
def run_sst_model(processor_stt, model_stt, vad_detector):

    # Function for spliting audio into chunks if longer than 30 seconds
    def chunk_array(data, chunk_size):
        chunks = []
        num_chunks = len(data) // chunk_size
        remainder = len(data) % chunk_size

        for i in range(num_chunks):
            chunks.append(data[i * chunk_size : (i + 1) * chunk_size])

        if remainder > 0:
            chunks.append(data[num_chunks * chunk_size :])

        return chunks
    
    # Function for processing audio data
    def process_audio_stt(audio_data_list):

        start_time = time.time()
        
        audio_data = np.concatenate(audio_data_list, axis=0)
        input_features = processor_stt(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        # Generate token ids
        predicted_ids = model_stt.generate(input_features, max_length = 448)
        # Decode token ids to text
        transcription = processor_stt.batch_decode(predicted_ids, skip_special_tokens=True)

        print(f'Speech to text translation time: {time.time() - start_time}')

        return transcription

    # Set up PyAudio stream
    p = pyaudio.PyAudio()
    chunk_size = 512
    format = pyaudio.paFloat32
    channels = 1
    rate = 16000

    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size)

    audio_data_list = []

    # Counts the number of consecutive audio chunks that are silent (a chunk of 512 is roughly 0.032 seconds)
    vad_sentance_end_couter = 0
    
    while True:
        
        print('Listening...')
        data = stream.read(chunk_size, exception_on_overflow = False)

        # Append audio chunk to audio_data
        data_edited = np.frombuffer(data, dtype=np.float32)
        audio_data_list.append(data_edited)

        vad_sentance_end_bool = run_vad(vad_detector, data)

        if vad_sentance_end_bool:

            vad_sentance_end_couter += 1

        else:
            vad_sentance_end_couter = 0

        if vad_sentance_end_couter > 60:

            print('Question end detected')
            print('Translating speech to text...')
            max_chunk_size = 850

            if len(audio_data_list) < max_chunk_size:
  
                transcription = process_audio_stt(audio_data_list)

            else:

                print('Chunking audio data...')
                transcription_list = []
                chunked_data = chunk_array(audio_data_list, max_chunk_size)

                for chunk in chunked_data:

                    transcription_part = process_audio_stt(chunk)
                    transcription_list.append(transcription_part[0])
                
                transcription = " ".join(transcription_list)
                transcription = [transcription]

            return transcription, vad_sentance_end_couter

# Run llm function
def run_llm(conversation, text_translation):

    print(f'Question: {text_translation}')

    start_time = time.time()
    
    llm_out = conversation.predict(input=text_translation)

    print(f'Time elapsed: {time.time() - start_time}')
    print(f'LLM Answer: {llm_out}')
    
    speak(engine, llm_out)
   
 # Speak back to you
def speak(engine, text):
    
    start_time = time.time()
     
    engine.say(text)
    engine.runAndWait ()

    print(f'Time elapsed: {time.time() - start_time}')

# Load bell audio files
def load_audio(file_path):
    """Load audio data from a WAV file."""
    with wave.open(file_path, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
    return audio_data, sample_width, channels, sample_rate

# Play bell audio files
def play_audio(audio_data, sample_width, channels, sample_rate):
    """Play audio data using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True)
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":

    #
    # Settings
    #
    which_llm = "mistral_3gb" #"stablelm" "quiklang", mistral_3gb
    acc_device = "cuda" # "cuda" "cpu"
    wake_word = "jarvis" # "jarvis", 'hey siri', 'hey google', 'terminator', 'alexa', 'ok google', 'computer'
    wake_word_sensitivity = 0.9 # 0.8
    voice_gender = 0 # 0 = male, 1 = female

    #
    # load acitivty detection
    #

    
    vad_file_path = os.path.join("/app/vad/silero_vad.onnx")

    print("Loading voice acitivty detection...")
    vad_detector = VADDetector2(vad_file_path)
    jarvis_handle = pvporcupine.create(keywords=[wake_word], sensitivities=[wake_word_sensitivity])

    #
    # Load speech to text
    #
    print("Loading voice understanding...")
    stt_file_path = '/app/sst/'
    
    processor_stt = WhisperProcessor.from_pretrained(stt_file_path, local_files_only = True)
    model_stt = WhisperForConditionalGeneration.from_pretrained(stt_file_path, local_files_only = True)

    #
    # load bell audio files
    #
    audio_file_path1 = "app/audio/bell_short_wav2.wav"
    audio_file_path2 = "app/audio/bell2_short_wav2.wav"
    audio_data, sample_width, channels, sample_rate = load_audio(audio_file_path1)
    audio_data2, sample_width2, channels2, sample_rate2 = load_audio(audio_file_path2)

    #
    # Load language model prompt and langchain memory
    #
    print("Loading language module...")

    template = """

    # Conversation history:
    # {history} - Conversation history end. 
    # You are an AI named Jarvis who is fun, and very intelligent.
    # Your goal is to help the human, answer questions, and bring humour.
    # Only speak in the first person as Jarvis, do not speak in prose, do not summarise the conversation, do not use smiley faces or emojis. Respond concisely.
    <|user|>
    {input}<|endoftext|>
    <|Jarvis|>
    """

    # Initialse prompt
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    # Store all conversation
    #memory = ConversationBufferMemory(ai_prefix="Jarvis")    
    
    # Store last K conversations
    memory = ConversationBufferWindowMemory(ai_prefix="Jarvis",k=10)

    llm_mapping = {
        "mistral_3gb": "mistral-7b-instruct-v0.1.Q2_K.gguf",
        "stablelm": "stablelm-zephyr-3b.Q4_K_M.gguf",
        "quiklang": "zephyr-quiklang-3b-4k.Q4_K_M.gguf"
    }

    source_llm = source_dir + 'llm_model/llm_model/' + llm_mapping.get(which_llm, "")   
   
    # LLM settings for creativity
    # Test: temperature = 0.7,    top_p=20,     top_k=0.4
    # Precise: temperature =0.7, top_k = 40, top_p = 0.1
    # Creative: temperature = 0.72, top_k= 0, top_p = 0.73
    # Sphinx: temperature = 1.99, top_k= 30, top_p = 0.18

    #
    # Load LLM
    #
    llm = LlamaCpp(
        model_path=source_llm,
        n_gpu_layers=-1,
        n_batch=124,
        n_ctx=2048,
        max_tokens=1024,
        verbose=True,  # Verbose is required to pass to the callback manager
        repetition_penalty = 1.6,
        temperature =0.75, top_k = 20, top_p = 0.4
    )   

    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory,
    )

    #
    # Load voice
    #
    print("Loading voice...")
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty ('voices')
    engine.setProperty ('voice', voices [voice_gender].id)

    print("Models loaded.")

    speak(engine, "Jarvis loaded, I am ready to assist you.")

    while True:
        try:
            
            # Detect wake word
            detect_wake_word(jarvis_handle)
            
            if wake_word_bool:

                print("Wake word detected!")

                #Sound 1
                play_audio(audio_data, sample_width, channels, sample_rate)
                
                # Run STT
                text_translation, vad_sentance_end_couter = run_sst_model(processor_stt, model_stt, vad_detector)
             
                #Sound 2
                play_audio(audio_data2, sample_width2, channels2, sample_rate2)

                # Run LLM
                if len(text_translation[0]) < 8:
                    print('Error: question too short')
                    continue
                
                else:
                    run_llm(conversation, text_translation)

        except KeyboardInterrupt:
            print("Ctrl+C detected. Exiting loop.")
            break
