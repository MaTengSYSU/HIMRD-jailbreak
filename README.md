<div align="center">

# 🗡️ HIMRD: Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models ICCV 2025

</div>

Code repository for the paper "Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models". 

- arXiv link: https://arxiv.org/abs/2412.05934

## Abstract

With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs.

Warning: This paper contains offensive and harmful examples, reader discretion is advised.

## Quick Start 

### Requirements

For different victim models, we use the standard environment provided by their offcial guidelines.
- deepseek-vl-7b-chat:      https://github.com/deepseek-ai/DeepSeek-VL
- llava-v1.5-7b:            https://github.com/haotian-liu/LLaVA
- llava-v1.6-mistral-7b-hf: https://github.com/haotian-liu/LLaVA
- glm-4v-9b:                https://github.com/THUDM/GLM-4
- MiniGPT-4:                https://github.com/Vision-CAIR/MiniGPT-4
- Qwen-VL-Chat:             https://github.com/QwenLM/Qwen-VL
- Yi-VL-34B:                https://github.com/01-ai/Yi

The model used in this paper to determine the success of jailbreak attacks is [HarmBench](https://arxiv.org/abs/2402.04249), which can be used in all of the above environments.

### Generate initial data

Firstly, we need to open `./data/model.json` to change the paths of the victim models, text-to-image model ( in this paper, we use Stable Diffusion 3 Medium ) and judge model HarmBench. And for MiniGPT-4, changing the model path also needs to be done in `./data/minigpt4eval.yaml`.

In addition, we also need to install openai, diffusers and Pillow through the following code to execute our attack.

Then, we need to set OpenAI's api_key:

```
pip install openai==0.28.0
pip install diffusers==0.32.1
pip install pillow==9.4.0
export OPENAI_API_KEY=your OpenAI’s api_key
```

Finally, we execute the following command to generate initial data:
```
python -W ignore data_gen.py \
  --attack_target_path "./data/SafeBench/SafeBench.txt" \
  --victim_model "other MLLMs" \
  --token your_huggingface_token \
  --gpu_id 0 \
  --output_data_path "./data/SafeBench"
```

Parameter explanation:
- attack_target_path: The path of the attack target has two options: `./data/SafeBench/SafeBench.txt` and `./data/tiny_SafeBench/tiny_SafeBench.txt`.
- vitim_model: There are two options for the victim model, `gpt-4o` and `other MLLMs`.
- token: Using the T2I model requires a huggingface token.
- gpu_id: Which GPU is used for model loading.
- output_data_path: The path of generated text input and image input.


### HIMRD Attack

Firstly, we need to set OpenAI's api_key using the following command:

```
export OPENAI_API_KEY=your OpenAI’s api_key
```

Then, we execute the following command to implement HIMRD attack：

```
python -W ignore attack.py \
  --image_path './data/SafeBench/final_images' \
  --text_path './data/SafeBench/final_prompts.txt' \
  --attack_target_path './data/SafeBench/SafeBench.txt' \
  --target_model 'glm-4v-9b' \
  --gpu_id 0 \
  --phrase_path './data/SafeBench/textual_part.txt' \
  --output_path './attack_result'
```

Parameter explanation:
- image_path: The path of generated image input.
- text_path: The path of generated text input.
- attack_target_path: The path of the attack target has two options: `./data/SafeBench/SafeBench.txt` and `./data/tiny_SafeBench/tiny_SafeBench.txt`.
- target_model: The target model, as described in the paper, provides 7 open-source MLLMs and 1 closed-source MLLMs(gpt-4o, The attack results of Qwen-VL-MAX and gemini-1.5-pro are implemented using AK and are not provided here.).
- gpu_id: Which GPU is used for model loading.
- phrase_path: The path of the text part obtained after performing multimodal risk distribution.
- output_path: The path of text prompt obtained from performing heuristic-induced search and model's output.
