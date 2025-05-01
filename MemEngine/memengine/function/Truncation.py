from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class BaseTruncation(ABC):
    """
    Helps to formulate memory contexts under the limitations of token number by certain LLMs.
    """
    def __init__(self, config) -> None:
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMTruncation(BaseTruncation):
    def __init__(self, config):
        super().__init__(config)

        if self.config.mode == 'token':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)

    def truncate_by_word(self, text):
        return ' '.join(text.split(' ')[:self.config.number])

    def truncate_by_token(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.config.number]

        truncated_text = self.tokenizer.convert_tokens_to_string(tokens)
        return truncated_text

    def get_piece_number(self, text):
        if self.config.mode == 'word':
            return len(text.split(' '))
        elif self.config.mode == 'token':
            return len(self.tokenizer.tokenize(text))
        else:
            raise "Truncation mode error."

    def check_truncation_needed(self, text):
        return self.get_piece_number(text) > self.config.number

    def __call__(self, text):
        if self.config.mode == 'word':
            return self.truncate_by_word(text)
        elif self.config.mode == 'token':
            return self.truncate_by_token(text)
        else:
            raise "Truncation mode error."