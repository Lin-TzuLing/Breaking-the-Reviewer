# from .model import OpenAIModelWrapper
from .model_wrapper import MistralModelWrapper



class MistralSmall3(MistralModelWrapper):
    """
    Mistral Small 3.1 model wrapper for paper review generation.
    This class provides a wrapper around the Mistral Small 3.1 model, specifically configured
    for generating reviews of academic papers. It inherits from MistralModelWrapper
    and sets default parameters optimized for paper review tasks.
    Args:
        model_name (str): The name of the Mistral Small 3.1 model to use.
        **kwargs: Additional keyword arguments including:
            port (int, optional): The port number for model communication.
    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate (default: 2048).
        model_temperature (float): Temperature parameter for response randomness (default: 0.3).
        system_prompt: System prompt configuration (default: None).
    Example:
        >>> mistral_small_3 = MistralSmall3("mistral-small-3-1", port=1234)
        >>> result = mistral_small_3("Evaluate the novelty of this paper...")
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
        Generate aspect review of papers.
        
        Args:   
            input_texts (:obj:`str`): the input prompt for predicting aspect review of papers.

        Returns:
            generated_text (:obj:`str`): the generated text.
            
        """

        generated_text = self.predict(input_texts)
        
        return generated_text