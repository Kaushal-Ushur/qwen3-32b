import os
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams
import huggingface_hub

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    huggingface_hub.login(hf_token)

class InferlessPythonModel:
    def initialize(self):
        model_name = "Qwen/Qwen3-14B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, enforce_eager=True)

    def infer(self, inputs):
        prompt = inputs["prompt"]
        # temperature = inputs.get("temperature", 0.6)
        top_p = inputs.get("top_p", 0.95)
        top_k = int(inputs.get("top_k", 20))
        repetition_penalty = float(inputs.get("repetition_penalty", 1.18))
        max_new_tokens = inputs.get("max_new_tokens", 256)

        # Apply chat template with reasoning enabled
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enables reasoning mode
        )

        sampling_params = SamplingParams(
            # temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens
        )

        result = self.llm.generate([formatted_prompt], sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"generated_text": result_output[0]}

    def finalize(self):
        self.llm = None
        self.tokenizer = None
