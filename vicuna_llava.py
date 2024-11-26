from transformers import AutoModelForCausalLM, LlamaForCausalLM, PreTrainedModel, CLIPImageProcessor, CLIPVisionModel, AutoTokenizer
from accelerate import Accelerator
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
from torch import Tensor
import json
import os
from torchvision.io import read_image
import numpy as np

class vicuna_llava(LlamaForCausalLM):
    def __init__(self, config, llmURL="lmsys/vicuna-7b-v1.5", clipvisionmodelURL="openai/clip-vit-large-patch14", accelerator=Accelerator()):
        super().__init__(config)

        self.model = AutoModelForCausalLM.from_pretrained(llmURL).to(self.device)
        self.accelerator = accelerator
        self.im_processor = self.accelerator.prepare(CLIPImageProcessor.from_pretrained(clipvisionmodelURL))
        self.clipvisionmodel = self.accelerator.prepare(CLIPVisionModel.from_pretrained(clipvisionmodelURL))
        self.tokenizer = self.accelerator.prepare(AutoTokenizer.from_pretrained(llmURL))
        self.special_tokens_dict = {"additional_special_tokens": ["<image>", "###", "/n", "<Human>", "<Assistant>"], "pad_token":"[PAD]"}
        self.num_added_tokens = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        if self.num_added_tokens > 0:
            with torch.no_grad():
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.vocab_size = len(self.tokenizer)

        self.system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        # pull LLM's word embedding layer out so prompt can be concatenated with im_embedding output
        self.pr_embedding = accelerator.prepare(self.model.get_input_embeddings())
        # construct embedding layer for image for projecting to LLM's word embedding layer
        self.im_embedding = accelerator.prepare(nn.Linear(1024, self.pr_embedding.embedding_dim))
        # replace LLM's embedding with identity since we will be using it outside the LLM
        self.model.set_input_embeddings(nn.Identity())



    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt',
                                        padding='longest',
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True).to(self.device)



    def vision_tower(self, image):
        proc_im = self.im_processor(image, return_tensors='pt').to(self.device)
        return self.clipvisionmodel(**proc_im).last_hidden_state.to(self.device)



    def process_prompt(self, prompt):
        
        # eos_token will be appended to prompt if we are training, otherwise we leave it
        eos_token = self.tokenizer.eos_token if self.training else ''
        if type(prompt) is list:
            for i in range(len(prompt)):
                prompt[i] = prompt[i].replace('<image>','').replace('/n','')
                prompt[i] = self.system_message + '<Human>' + prompt[i] + eos_token
        else:
            prompt = prompt.replace('<image>','').replace('/n','')
            prompt = self.system_message + '<Human>' + prompt + '<Assistant>'

        return prompt
        


    def forward(self, image:Tensor, prompt:str, batch_size=1):

        # tokenize prompt and extract image features
        with torch.no_grad():
            tokenized_pr = self.tokenize(prompt).to(self.device)
            clip_encoded_im = self.vision_tower(image).to(self.device)
            tokenized_pr_input_ids = tokenized_pr['input_ids'].to(self.device)
            tokenized_pr_attention_mask = tokenized_pr['attention_mask'].to(self.device)

        # project input image to LLM's embedding dimension
        projected_im = self.im_embedding(clip_encoded_im).to(self.device)
        embedded_pr = self.pr_embedding(tokenized_pr_input_ids).to(self.device)

        # concatenate tokenized prompt and image features, place start token embedded_pr[:,[0],:] at beginning of context window 
        embedded_im_and_pr = torch.cat((embedded_pr[:,[0],:],
                                        projected_im, 
                                        embedded_pr[:,1:,:]), dim=1).to(self.device)
        im_and_pr_attention_mask = torch.cat((tokenized_pr_attention_mask[:,[0]], 
                                            torch.ones(batch_size,projected_im.size()[1]).to(self.device), 
                                            tokenized_pr_attention_mask[:,1:]),dim=1).to(self.device)
            
        # send inputs through LLM, apply softmax to make outputs more clear
        llama_output = self.model.forward(input_ids=embedded_im_and_pr,attention_mask=im_and_pr_attention_mask)
        llama_output = torch.softmax(llama_output.logits, dim=2).to(self.device)

        # if we are training, compute causal language modeling loss
        if self.training:
            separator_token_id = self.tokenizer.convert_tokens_to_ids("###")
            loss_fn = self.accelerator.prepare(nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
            with autocast():
                # shift outputs and inputs for causal loss calculation
                shifted_outs = llama_output[:,:-1,:].contiguous()
                shifted_labels = tokenized_pr_input_ids[:,1:].contiguous()

                # mask image and input prompt out of loss computation
                im_loss_mask = torch.full_like(projected_im[:,:,0], -100)
                labels = shifted_labels.clone()
                labels[(labels == separator_token_id).cumsum(dim=1) == 0] = -100
                lossmask = torch.cat((im_loss_mask, labels), dim=1).type(torch.LongTensor).to(self.device)

                # compute loss, average loss over predicted response
                loss = loss_fn(shifted_outs.view(-1,self.model.config.vocab_size), lossmask.view(-1)).to(self.device)
                loss_per_sample = loss[loss!=0].mean()

            return {'outs':llama_output.to(self.device), 'loss':loss_per_sample}
        else:
            return llama_output.to(self.device)
    


    def generate(self, image:Tensor, prompt:str, max_new_tokens=50, batch_size=1):
        self.training=False
        num_tokens = 0
        pred_tokens = ''
        eos_token = self.tokenizer.special_tokens_map['eos_token']
        processed_prompt = self.process_prompt(prompt)
        while num_tokens < max_new_tokens:
            outs = self.forward(image,processed_prompt+pred_tokens,batch_size=1).to(self.device)

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