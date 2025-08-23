# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import json
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class LLMModelWrapper(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    port: int
        The port number for the server (default is 8091).

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """

    def __init__(
        self, model_name, max_new_tokens, temperature, port=None, device="auto"
    ):
        model_name_mapping = {
            "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
            "Mistral-small-3.1": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        }
        model_name = (
            model_name_mapping[model_name]
            if model_name in model_name_mapping
            else model_name
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.port = port

    @abstractmethod
    def predict(self, input_text, **kwargs):
        raise NotImplementedError

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class OpenAIModelWrapper(LLMModelWrapper):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """

    def __init__(
        self, model_name, max_new_tokens, temperature, system_prompt, openai_key, n=1
    ):
        super().__init__(model_name, max_new_tokens, temperature)

        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=openai_key)
        self.n = n

    async def predict(self, input_text):
        result = []

        # Gather the results concurrently using asyncio
        tasks = [self.get_query(text) for text in input_text]

        # Run all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, Exception):
                print(f"An error occurred: {response}")
            else:
                result.append(response)

        return result

    async def get_query(self, text, retries=500, delay=5):
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    n=self.n,
                    timeout=120,
                )
                if self.n > 1:
                    return [choice["message"]["content"] for choice in response.choices]
                else:
                    return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"Error: {error_msg}, attempt {attempt}")

                # resolve rate limit errors
                if "rate_limit" in error_msg or "429" in error_msg:
                    try:
                        error_data = json.loads(error_msg[error_msg.find("{") :])
                        wait_time = float(
                            error_data["error"]["message"]
                            .split("try again in ")[1]
                            .split("s")[0]
                        )
                    except Exception:
                        wait_time = delay  # fallback
                    await asyncio.sleep(wait_time)
                elif attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    return f"Error: {error_msg}"
                    exit()


class LlamaModelWrapper(LLMModelWrapper):
    """
    Language model class for interfacing with Llama models.

    Inherits from LMMBaseModel and sets up a model interface for Llama models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).
    port: int
        The port number for the server (default is 8091).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the Llama model.
    """

    def __init__(
        self, model_name, max_new_tokens, temperature, system_prompt, port, n=1
    ):
        super().__init__(model_name, max_new_tokens, temperature, port)

        base_url = f"http://localhost:{self.port}/v1"
        key = "abc"

        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=key, base_url=base_url)
        self.n = n

    async def predict(self, input_text):
        result = []

        # Gather the results concurrently using asyncio
        tasks = [self.get_query(text) for text in input_text]

        # Run all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, Exception):
                print(f"An error occurred: {response}")
            else:
                result.append(response)

        return result

    async def get_query(self, text, retries=80, delay=0.1):
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    n=self.n,
                    timeout=120,
                )
                if self.n > 1:
                    return [choice["message"]["content"] for choice in response.choices]
                else:
                    return response.choices[0].message.content

            except Exception as e:
                if attempt < retries - 1:
                    print(f"Error: {str(e)}, attempt {attempt}")
                    await asyncio.sleep(delay)  # wait before retrying
                else:
                    return f"Error: {str(e)}"


class MistralModelWrapper(LLMModelWrapper):
    """
    Language model class for interfacing with Mistral models.

    Inherits from LMMBaseModel and sets up a model interface for Mistral models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).
    port: int
        The port number for the server (default is 8091).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the Mistral model.
    """

    def __init__(
        self, model_name, max_new_tokens, temperature, system_prompt, port, n=1
    ):
        super().__init__(model_name, max_new_tokens, temperature, port)

        base_url = f"http://localhost:{self.port}/v1"
        key = "abc"

        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=key, base_url=base_url)
        self.n = n

    async def predict(self, input_text):
        result = []

        # Gather the results concurrently using asyncio
        tasks = [self.get_query(text) for text in input_text]

        # Run all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, Exception):
                print(f"An error occurred: {response}")
            else:
                result.append(response)

        return result

    async def get_query(self, text, retries=80, delay=0.1):
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    n=self.n,
                    timeout=120,
                )
                if self.n > 1:
                    return [choice["message"]["content"] for choice in response.choices]
                else:
                    return response.choices[0].message.content

            except Exception as e:
                if attempt < retries - 1:
                    print(f"Error: {str(e)}, attempt {attempt}")
                    await asyncio.sleep(delay)  # wait before retrying
                else:
                    return f"Error: {str(e)}"
