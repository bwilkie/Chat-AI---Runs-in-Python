# Chat AI, Runs locally in Python

For anyone interested in talking to an AI Large Language Model in real time. (See my other Chat-AI Colab repo for an easy to use variation). 


This implementation has the following features;

1) Voice activity detection
2) Wake Word Recognition
3) Speech to Text
4) A Large Language Model
5) Text to Speech

This one requires dependancies but is a great starting point to build new features. 

1) Porcupine for Wake Word detection is required to be pvporcupine==1.9.5 as this required no Auth Keys
2) Gguf are lightweight, fast llm files, and require LlamaCpp. This may require other dependancies such as c++. You can use non gguf files but you will need to replace LlamaCpp with transformers. 
3) Models for LLM and VAD will need to be downloaded and put in the required directory. Im also using some audio files for interaction. I will link these below. (this uses gguf files for speed, recommend the-bloke repo on hugging face)

TheBloke and Mistral (Any gguf will work)
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
Choose a model, i.e. mistral-7b-instruct-v0.1.Q2_K.gguf

Snakers4 and the file for Voice Activity Detection
https://github.com/snakers4/silero-vad/tree/master/files/silero_vad.onnx

***** NOTES *****
You  can select some adjustments after __main__

1) Exit the program by clicking in the terminal and pressing Ctrl + C
2) You may need to change "cuda" to "cpu" if your not working with NVidia hardware or dont have cuda installed

-----------

If anyone would like to work on building this into a docker container with a web UI please hit me up. 



