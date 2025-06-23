# krknctl-lightspeed
## Overview
This repository offers tools for fine-tuning a Code-llama 7b model. 
The goal is to enable the [krknctl](https://github.com/krkn-chaos/krknctl)  command-line interface to generate commands
directly from natural language input, similar to how Ansible Lightspeed works.

## Requirements
- An Apple Silicon Mac (M series)
- [Ollama](https://ollama.com/) installed 

## Model Training

### Contributing to `krknctl-lightspeed-codellama` training data


We're aiming to make it easy for anyone to improve the `krknctl-lightspeed-codellama` model's accuracy. 
A key goal of this project is to create a format for expanding the training data that's both **readable and simple to compile**. 
This will allow more people to contribute by adding new `krknctl` use cases.

To contribute a use case, just edit the **`meta_commands.json` file**. This file is organized by **scenario**, with each scenario containing:

* The expected impact of the scenario on the cluster.
* A dictionary of the `krknctl` parameters that were configured

```
{
  "application-outages": [
    {
      "prompt": "Make the 'frontend-web' application unreachable in the 'prod' namespace for 90 seconds, blocking both Ingress and Egress traffic using pod selector '{app: frontend}'.",
      "params": {
        "duration": 90,
        "namespace": "prod",
        "block-traffic": "[Ingress, Egress]",
        "pod-selector": "{app: frontend}"
      }
    }
 ...
 ],
 ...
 }
```

These objects will be translated by the script into JSONL training data to the model automatically before starting the training. 

### Launch Training

Create a venv and install all the dependencies:

```
python -m venv training
source training/bin/activate
pip install -r requirements.txt
```

After setting up all the necessary dependencies, you can launch the training script with:

```
python training.py
```
> [!IMPORTANT] 
> All the training parameters are optimized to run on Apple Silicon GPUs

>[!NOTE] 
> The training phase takes approximately 45 minutes on a Macbook PRO M3 Pro


### Converting the model to `gguf` format

---
To run the trained model in **Ollama**, you'll need to convert it to the Ollama **GGUF format**. You can do this using the `llama.cpp` tool.

Clone the llama.cpp.git and install all the dependencies:

> [!TIP] 
> If you're converting the model in the same terminal where you previously ran 
> training, **remember to deactivate the training environment** using 
> the `deactivate` command before creating and activating a new virtual environment.


````
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
python -m venv conversion
source conversion/bin/activate
pip install -r requirements.txt
````

Run the model conversion from the llama.cpp folder with:

```
python3 convert_hf_to_gguf.py <krknctl_lightspeed_path>/krknctl-lightspeed-codellama \
    --outtype f16 \
    --outfile <krknctl_lightspeed_path>/krknctl-lightspeed-codellama.gguf
```

Install the freshly converted model in ollama with:

```
ollama create krknctl-lightspeed -f <krknctl_lightspeed_path>/Modelfile
```

If everything is ok you should be now able to run and chat `krknctl-lightspeed`  with:

```
ollama run krknctl-lightspeed
```


[![asciicast](https://asciinema.org/a/RC7WRTGZKnM29rqTBL2wGCfZm.svg)](https://asciinema.org/a/RC7WRTGZKnM29rqTBL2wGCfZm)

