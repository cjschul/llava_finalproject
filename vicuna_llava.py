from transformers import LlamaModel, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn as nn
import torch

class vicuna_llava(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.im_embedding = nn.Linear(768, 4096)
        self.pr_embedding = self.model.get_input_embeddings()
        self.model.set_input_embeddings(nn.Identity())

    def forward(self, clip_encoded_im : BaseModelOutputWithPooling, tokenized_pr: BatchEncoding):
        tokenized_pr_input_ids = tokenized_pr['input_ids']
        tokenized_pr_attention_mask = tokenized_pr['attention_mask']

        projected_im = self.im_embedding(clip_encoded_im['last_hidden_state'])
        embedded_pr = self.pr_embedding(tokenized_pr_input_ids)

        im_and_pr_input_ids = torch.cat((projected_im, embedded_pr), dim=1)
        im_and_pr_attention_mask = torch.cat((torch.ones(1,projected_im.size()[1]), tokenized_pr_attention_mask),dim=1)

        return self.model.forward(input_ids=im_and_pr_input_ids,attention_mask=im_and_pr_attention_mask)            
        