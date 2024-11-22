from transformers import AutoModelForCausalLM, LlamaForCausalLM, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import json
import os
from torchvision.io import read_image

class vicuna_llava(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # grab pretrained vicuna-7B-v1.5 for main LM functionality
        self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

        # construct embedding layer for image for projecting to vicuna's word embedding layer
        self.im_embedding = nn.Linear(768, 4096)

        # pull vicuna's word embedding layer out so prompt can be concatenated with im_embedding output
        self.pr_embedding = self.model.get_input_embeddings()

        # replace vicuna's embedding with identity
        self.model.set_input_embeddings(nn.Identity())
        self.lm_head = self.model.lm_head

    def forward(self, clip_encoded_im:BaseModelOutputWithPooling, 
                tokenized_pr:BatchEncoding,
                batch_size=None):
        
        # grab token ids, attention mask from input prompts
        tokenized_pr_input_ids = tokenized_pr['input_ids']
        tokenized_pr_attention_mask = tokenized_pr['attention_mask']

        # project input image to vicuna v1.5 7B's word embedding layer
        projected_im = self.im_embedding(clip_encoded_im['last_hidden_state'])
        embedded_pr = self.pr_embedding(tokenized_pr_input_ids)

        # concatenate our embedded prompt and projected image features
        # also place the start token at front (embedded_pr[:,[0],:])
        embedded_im_and_pr = torch.cat((embedded_pr[:,[0],:],
                                        projected_im, 
                                        embedded_pr[:,1:,:]), dim=1)

        # concatenate attention masks
        im_and_pr_attention_mask = torch.cat((tokenized_pr_attention_mask[:,[0]], 
                                            torch.ones(batch_size,projected_im.size()[1]), 
                                            tokenized_pr_attention_mask[:,1:]),dim=1)
            
        
        # call  vicuna on concatenated image tokens + prompt tokens + response tokens
        llama_output = self.model.forward(input_ids=embedded_im_and_pr,attention_mask=im_and_pr_attention_mask)

        # apply softmax on outputs
        llama_output = torch.softmax(llama_output.logits, dim=1)

        return llama_output

class dataset_llava(Dataset):
    def __init__(self, prompts_file:str, images_dir:str):
        with open(prompts_file, 'r') as file:
            self.prompts = json.load(file)
        self.images_dir = images_dir
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir)
        img_name = self.prompts[idx]['image']
        image = read_image(img_path+img_name)
        prompt = self.prompts[idx]['conversations'][0]['value']
        resp = self.prompts[idx]['conversations'][1]['value']


        return prompt, image, resp