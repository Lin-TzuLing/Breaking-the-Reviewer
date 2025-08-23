# from .model import OpenAIModelWrapper
from .model_wrapper import LlamaModelWrapper



class Llama70B(LlamaModelWrapper):
    """
    Llama 70B model wrapper for paper review generation.
    This class extends LlamaModelWrapper to provide a specific implementation
    for the Llama 70B model with predefined configuration parameters including
    temperature, token limits, and system prompt settings.
    Args:
        model_name (str): The name/path of the Llama 70B model to load.
        **kwargs: Additional keyword arguments.
            port (int, optional): The port number for model communication. 
                                Defaults to None.
    Attributes:
        system_prompt (None): System prompt configuration, set to None by default.
        max_new_tokens (int): Maximum number of tokens to generate, set to 2048.
        model_temperature (float): Sampling temperature for text generation, set to 0.3.
    Example:
        >>> model = Llama70B("path/to/llama70b/model")
        >>> result = model("Evaluate this research paper...")
        >>> print(result)
    """
    
    def __init__(self, model_name, **kwargs):
        system_prompt = None
        max_new_tokens = 2048
        model_temperature = 0.3
        port = kwargs.get('port', None)
        super().__init__(model_name, max_new_tokens, model_temperature, system_prompt, port)   
        

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