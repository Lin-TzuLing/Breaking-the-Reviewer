from .model_wrapper import OpenAIModelWrapper


class GPT4oMini(OpenAIModelWrapper):
    """
    GPT-4o-mini model wrapper for paper review generation.
    This class provides a wrapper around the OpenAI GPT-4o-mini model, specifically configured
    for generating reviews of academic papers. It inherits from OpenAIModelWrapper
    and sets default parameters optimized for paper review tasks.
    Args:
        model_name (str): The name of the GPT-4o-mini model to use.
        **kwargs: Additional keyword arguments including:
            openai_key (str, optional): OpenAI API key for authentication.
    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate (default: 2048).
        model_temperature (float): Temperature parameter for response randomness (default: 0.3).
        system_prompt: System prompt configuration (default: None).
    Example:
        >>> gpt4o_mini = GPT4oMini("gpt-4o-mini", openai_key="your-api-key")
        >>> result = gpt4o_mini("Evaluate the novelty of this paper...")
        >>> print(result)
    """
    
    def __init__(self, model_name, **kwargs):
        openai_key = kwargs["openai_key"] if "openai_key" in kwargs else None
        system_prompt = None
        max_new_tokens = 2048
        model_temperature = 0.3
        super().__init__(
            model_name, max_new_tokens, model_temperature, system_prompt, openai_key
        )

    def __call__(self, input_texts):
        """
        Generate review of papers.

        Args:
            input_texts (:obj:`str`): the input prompt for predicting review of papers.

        Returns:
            generated_text (:obj:`str`): the generated text.

        """

        generated_text = self.predict(input_texts)

        return generated_text
