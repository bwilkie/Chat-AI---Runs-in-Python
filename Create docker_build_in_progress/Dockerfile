FROM python:3.9


RUN apt update && apt upgrade -y

#FROM dockcross/manylinux-x64
# RUN yum update -y
# #RUN yum install -y sudo

# RUN yum install rh-python38 -y
# RUN yum install python3-pip -y
# RUN yum install cmake -y

RUN apt install python3-pip -y

RUN apt-get install portaudio19-dev -y

RUN pip install numpy pyaudio
RUN pip install onnxruntime transformers pyttsx3 langchain wave
RUN pip install pvporcupine==1.9.5
RUN pip install torch==2.0.1
RUN pip install comtypes
RUN apt install espeak ffmpeg libespeak1 -y
RUN pip install llama-cpp-python

#RUN apt-get install -y alsa-base alsa-utils

# # RUN pip3 install flask joblib pandas scikit-learn opencv-python 
# ENV LC_ALL=en_US.utf-8
# ENV LANG=en_US.utf-8

# Copy vad
COPY files/silero_vad.onnx /app/vad/silero_vad.onnx

# Copy stt
COPY files/sst_model/ /app/sst/

# Copy audio
COPY files/bell_short_wav2.wav app/audio/bell_short_wav2.wav
COPY files/bell2_short_wav2.wav app/audio/bell2_short_wav2.wav

# Copy LLM
#COPY D:/ai_models/llm_model/llm_model/mistral-7b-instruct-v0.1.Q2_K.gguf /app/llm/mistral-7b-instruct-v0.1.Q2_K.gguf
COPY files/mistral-7b-instruct-v0.1.Q2_K.gguf /app/llm/mistral-7b-instruct-v0.1.Q2_K.gguf

#COPY   D:/ai_models/llm_model/llm_model/zephyr-quiklang-3b-4k.Q4_K_M.gguf /app/llm/zephyr-quiklang-3b-4k.Q4_K_M.gguf

COPY model.py model.py

#RUN python3 model.py
#ENTRYPOINT FLASK_APP=flaskapp.py flask run --host=0.0.0.0 --port=80
# ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=80"]
CMD ["python3", "model.py"]
