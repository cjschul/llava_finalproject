from transformers import AutoModelForCausalLM, LlamaForCausalLM, PreTrainedModel, CLIPImageProcessor, CLIPVisionModel, AutoTokenizer
from accelerate import Accelerator
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch import Tensor
import json
import os
from torchvision.io import read_image
import numpy as np

class vicuna_llava(LlamaForCausalLM):
    def __init__(self, config, llmURL="lmsys/vicuna-7b-v1.5", clipvisionmodelURL="openai/clip-vit-large-patch14", accelerator=Accelerator()):
        super().__init__(config)
        # grab pretrained vicuna-7B-v1.5 for main LM functionality
        self.model = AutoModelForCausalLM.from_pretrained(llmURL).to(self.device)

        # vision tower
        self.im_processor = accelerator.prepare(CLIPImageProcessor.from_pretrained(clipvisionmodelURL))
        self.clipvisionmodel = accelerator.prepare(CLIPVisionModel.from_pretrained(clipvisionmodelURL))

        # tokenizer
        self.tokenizer = accelerator.prepare(AutoTokenizer.from_pretrained(llmURL))
        self.special_tokens_dict = {"additional_special_tokens": ["<image>", "###", "/n"],
                                    "pad_token":"[PAD]"}
        self.num_added_tokens = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        if self.num_added_tokens > 0:
            with torch.no_grad():
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.vocab_size = len(self.tokenizer)


        # pull vicuna's word embedding layer out so prompt can be concatenated with im_embedding output
        self.pr_embedding = accelerator.prepare(self.model.get_input_embeddings())
        # construct embedding layer for image for projecting to vicuna's word embedding layer
        self.im_embedding = accelerator.prepare(nn.Linear(1024, self.pr_embedding.embedding_dim))
        # replace vicuna's embedding with identity
        self.model.set_input_embeddings(nn.Identity())



    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt',
                                        padding='longest',
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True).to(self.device)

    def vision_tower(self, image):
        proc_im = self.im_processor(image, return_tensors='pt').to(self.device)
        return self.clipvisionmodel(**proc_im).last_hidden_state.to(self.device)

    def forward(self, image:Tensor, prompt:str, batch_size=1):
        with torch.no_grad():
            # tokenize prompt
            tokenized_pr = self.tokenize(prompt).to(self.device)

            # process image, extract features
            clip_encoded_im = self.vision_tower(image).to(self.device)
    
        # grab token ids, attention mask from input prompts
        tokenized_pr_input_ids = tokenized_pr['input_ids'].to(self.device)
        tokenized_pr_attention_mask = tokenized_pr['attention_mask'].to(self.device)

        # project input image to vicuna v1.5 7B's word embedding layer
        projected_im = self.im_embedding(clip_encoded_im).to(self.device)
        embedded_pr = self.pr_embedding(tokenized_pr_input_ids).to(self.device)

        # concatenate our embedded prompt and projected image features
        # also place the start token at front (embedded_pr[:,[0],:])
        embedded_im_and_pr = torch.cat((embedded_pr[:,[0],:],
                                        projected_im, 
                                        embedded_pr[:,1:,:]), dim=1).to(self.device)

        # concatenate attention masks
        im_and_pr_attention_mask = torch.cat((tokenized_pr_attention_mask[:,[0]], 
                                            torch.ones(batch_size,projected_im.size()[1]).to(self.device), 
                                            tokenized_pr_attention_mask[:,1:]),dim=1).to(self.device)
            
        
        # call  vicuna on concatenated image tokens + prompt tokens + response tokens
        llama_output = self.model.forward(input_ids=embedded_im_and_pr,attention_mask=im_and_pr_attention_mask)

        # apply softmax on outputs
        llama_output = torch.softmax(llama_output.logits, dim=1).to(self.device)

        return llama_output.to(self.device)
    
    def generate(self, image:Tensor, prompt:str, max_new_tokens=50, batch_size=1):
        num_tokens = 0
        pred_tokens = ''
        eos_token = self.tokenizer.special_tokens_map['eos_token']

        while num_tokens < max_new_tokens:
            outs = self.forward(image,prompt+pred_tokens,batch_size=1).to(self.device)

            pred_token_id = torch.argmax(outs[:,-1,:]).to(self.device)
            pred_token = self.tokenizer.decode(pred_token_id)

            pred_tokens+=pred_token
            if pred_token_id == eos_token:
                break

            num_tokens+=1

        return pred_tokens





class dataset_llava(Dataset):
    def __init__(self, prompts_file:str, images_dir:str):
        super().__init__()
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