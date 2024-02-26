# Chat-AI---Runs-in-Python

Real time chat based LLM designed to run on windows 

For anyone interested in talking to an AI Large Language Model in real time.

See my other Chat-AI Colab repo for an easy to use variation. This requires dependancies but is a great starting point to build new features. 

This implementation has the following features;

1) Voice activity detection
2) Wake Word Recognition
3) Speech to Text
4) A Large Language Model
5) Text to Speech


***** NOTEs
You  can select some adjustments after __main__

1) Exit the program by clicking in the terminal and pressing Ctrl + C
2) Porcupine for Wake Word detection is required to be pvporcupine==1.9.5 as this required no Auth Keys
3) Models for LLM and VAD will need to be downloaded and put in the required directory
4) gguf files if I remeber correctly require a C++ engine. As an alternative, Refact model is also loaded in code and can be selection
5) You may need to change "cuda" to "cpu" if your not working with NVidia hardware or dont have cuda installed

Download requirments and thanks;

Snakers4 and the file for Voice Activity Detection
https://github.com/snakers4/silero-vad/tree/master/files/silero_vad.onnx

TheBloke and Mistral
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
Choose a model, i.e. mistral-7b-instruct-v0.1.Q2_K.gguf

OR

Refact 
https://huggingface.co/smallcloudai/Refact-1_6B-fim/tree/main
