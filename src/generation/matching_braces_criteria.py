import torch
from transformers import StoppingCriteria

class MatchingBracesCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.
    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, tokenizer, input_seq):
        self.tokenizer = tokenizer
        self.cnt = 0
        self.cnt_max = 0
        self.scan(input_seq)
        if self.cnt == 0:
            self.cnt_max = 0

    def scan(self, seq):
        for char in seq:
            if char == "{":
                self.cnt += 1
            if char == "}":
                self.cnt -= 1
            self.cnt_max = max(self.cnt_max, self.cnt)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        token = input_ids[:,-1][0]
        seq = self.tokenizer.decode(token)
        self.scan(seq)
        
        if self.cnt == 0 and self.cnt_max > 0:
            return True
        return False
