# krknctl-lightspeed

# Training

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

pip install huggingface_hub[hf_transfer]

python llama.cpp/convert.py /path/to/your/krknctl_finetuned_model --outfile /path/to/your/krknctl-generator-model.gguf

ollama create krknctl-generator -f ./Modelfile

