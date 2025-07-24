import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchattacks
import random
from collections import defaultdict, deque
from torch import optim
import json
import matplotlib.pyplot as plt
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

class MiniNoTemplate():
    def __init__(self, args, model, model_name, tokenizer, prompt_num, train_batch_demo_samples, cropa_iter,
                 ques_id_to_img_id, specific_prompt, agnostic_prompt, image_prompt, cls_prompt, cap_prompt, loss_json,
                 language_loss_json, atten_loss_json, cos_json):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_num = prompt_num
        self.train_batch_demo_samples = train_batch_demo_samples
        self.cropa_iter = cropa_iter
        self.ques_id_to_img_id = ques_id_to_img_id
        self.specific_prompt = specific_prompt
        self.agnostic_prompt = agnostic_prompt
        self.image_prompt = image_prompt
        self.cls_prompt = cls_prompt
        self.cap_prompt = cap_prompt
        self.model_name = model_name
        self.loss_json = loss_json
        self.language_loss_json = language_loss_json
        self.atten_loss_json = atten_loss_json
        self.cos_json = cos_json


    def get_intended_token_ids(self, input_ids, target_id, debug=False):
        padding = torch.full_like(input_ids, -100)
        padding_dim = padding.shape[1]
        for i in range(len(target_id)):
            padding[:, padding_dim - len(target_id) + i] = target_id[i]
        if debug:
            print("input_ids is:", input_ids)
            print("target_id is:", target_id)
            print("padding is:", padding)
        return padding


    def attack(self, tpoch, image, question_id, target_text, steps, alpha1, epsilon, device):
        # 获取问题id对应的图像id
        image_id = str(self.ques_id_to_img_id[question_id])

        # 获取该图像的额外提示词
        specific_prompt = self.specific_prompt[image_id]
        image_prompt = self.image_prompt[image_id]
        agnostic_prompt = self.agnostic_prompt['train_agn']
        cls_prompt = self.cls_prompt['train_cls']
        cap_prompt = self.cap_prompt['train_cap']

        item_images = []
        item_text = []
        context_images = []
        item_images.append(context_images + [image])

        for ques in specific_prompt:
            item_text.append(ques)
        for ques in image_prompt:
            item_text.append(ques)
        for ques in agnostic_prompt:
            item_text.append(ques)
        for ques in cap_prompt:
            item_text.append(ques)
        for ques in cls_prompt:
            item_text.append(ques)

        text_list = []
        if self.model_name == 'blip2':
            for ques in item_text:
                text_list.append(f'Question: {ques} Answer: ')
            item_text = text_list
        elif self.model_name == 'llava':
            for ques in item_text:
                text_list.append(f"USER: <image>\n{ques} ASSISTANT:")
            item_text = text_list
        elif self.model_name == 'open_flamingo':
            train_context_text = "".join([
                self.model.get_vqa_prompt(
                    question=x["question"], answer=x["answers"][0]
                )
                for x in self.train_batch_demo_samples
            ])
            for ques in item_text:
                text_list.append(train_context_text + self.model.get_vqa_prompt(question=ques))
            item_text = text_list

        labels_list = []

        input_encodings = self.tokenizer(
            item_text, padding="longest",
            truncation=True, return_tensors="pt", max_length=2000)
        input_ids = input_encodings["input_ids"].to(device)
        attention_mask = input_encodings["attention_mask"].to(device)

        if self.model_name == 'instructblip':
            qformer_text_encoding = self.model.qformer_tokenizer(item_text, padding="longest",
                                                                 truncation=True, return_tensors="pt",
                                                                 max_length=2000).to(device)
            qformer_input_ids = qformer_text_encoding["input_ids"]
            qformer_attention_mask = qformer_text_encoding["attention_mask"]

        for k in range(len(item_text)):
            target_id = self.tokenizer.encode(target_text)[1:]
            # labels = torch.tensor([target_id]).to(device)
            labels = self.get_intended_token_ids(input_ids[k].unsqueeze(0), target_id)
            labels_list.append(labels)
        normalize_image = self.model._prepare_images_no_normalize(item_images).to(device)

        if self.args.MIM or self.args.NIM:
            momentum = torch.zeros_like(normalize_image).detach().to(device)

        labels = torch.cat(labels_list, dim=0).to(device)
        adv_image = normalize_image.clone().detach()

        for _ in range(steps):
            adv_image.requires_grad = True

            grad_inner_list = []
            loss_inner_list = []

            adv_image_inner = adv_image.clone().detach()

            for l in range(self.args.K * 65 // self.args.minibatch_num):
            # for l in range(4 * 65 // self.args.minibatch_num):
                adv_image_inner.requires_grad = True

                if self.args.NIM:
                    nes_images = adv_image_inner + 1.0 * alpha1 * momentum

                indices = random.sample(range(len(input_ids)), self.args.minibatch_num)

                input_ids_sample = torch.stack([input_ids[i] for i in indices], dim=0)
                attention_mask_sample = torch.stack([attention_mask[i] for i in indices], dim=0)

                labels_sample = torch.stack([labels[i] for i in indices], dim=0)

                if self.model_name == 'instructblip':
                    qformer_input_ids_sample = torch.stack([qformer_input_ids[i] for i in indices], dim=0)
                    qformer_attention_mask_sample = torch.stack([qformer_attention_mask[i] for i in indices], dim=0)
                    if self.args.NIM:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([nes_images] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True,
                            qformer_input_ids=qformer_input_ids_sample,
                            qformer_attention_mask=qformer_attention_mask_sample
                        )[0]
                    else:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([adv_image_inner] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True,
                            qformer_input_ids=qformer_input_ids_sample,
                            qformer_attention_mask=qformer_attention_mask_sample
                        )[0]
                elif self.model_name == 'blip2':
                    if self.args.NIM:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([nes_images] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True
                        )[0]
                    else:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([adv_image_inner] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True
                        )[0]
                elif self.model_name == 'llava':
                    if self.args.NIM:
                        loss = -1 * self.model.model(
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([nes_images] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True
                        )[0]
                    else:
                        loss = -1 * self.model.model(
                            input_ids=input_ids_sample,
                            pixel_values=torch.cat([adv_image_inner] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample,
                            normalize_vision_input=True
                        )[0]
                elif self.model_name == 'open_flamingo':
                    if self.args.NIM:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            lang_x=input_ids_sample,
                            vision_x=torch.cat([nes_images.half()] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample
                        )[0]
                    else:
                        loss = -1 * self.model.model(
                            inputs_embeds=None,
                            lang_x=input_ids_sample,
                            vision_x=torch.cat([adv_image_inner.half()] * self.args.minibatch_num, dim=0),
                            attention_mask=attention_mask_sample,
                            labels=labels_sample
                        )[0]

                grad_inner = torch.autograd.grad(loss, adv_image_inner, retain_graph=False, create_graph=False)[0]

                grad_inner = grad_inner / torch.mean(torch.abs(grad_inner), dim=(1, 2, 3), keepdim=True)

                adv_image_inner = adv_image_inner.detach() + alpha1 * grad_inner.sign()
                delta_inner = torch.clamp(adv_image_inner - normalize_image, min=-epsilon, max=epsilon)
                adv_image_inner = torch.clamp(normalize_image + delta_inner, min=0, max=1).detach()

                grad_inner_list.append(grad_inner)
                loss_inner_list.append(loss)

                torch.cuda.empty_cache()

            if _ == 0:
                if self.model_name == 'llava':
                    grad_ = torch.zeros((1, 3, 336, 336), dtype=torch.float32).to(device)
                elif self.model_name == 'open_flamingo':
                    grad_ = torch.zeros((1, 1, 1, 3, 224, 224), dtype=torch.float32).to(device)
                else:
                    grad_ = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)
            else:
                grad_ = grad_temp

            grad = torch.mean(torch.cat(grad_inner_list, dim=0).detach().clone(), dim=0, keepdim=True)

            if self.args.MIM or self.args.NIM:
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * 0.9
                momentum = grad

            grad_temp = copy.deepcopy(grad)
            cos_sim = F.cosine_similarity(grad.view(1, -1), grad_.view(1, -1))
            self.cos_json[image_id].append(float(cos_sim.item()))

            loss_record = -1 * torch.mean(torch.stack(loss_inner_list))
            self.loss_json[image_id].append(float(loss_record.item()))

            tpoch.set_postfix(loss=loss_record.item(), cos=cos_sim.item(), step=_)

            adv_image = adv_image.detach() + alpha1 * grad.sign()
            delta = torch.clamp(adv_image - normalize_image, min=-epsilon, max=epsilon)
            adv_image = torch.clamp(normalize_image + delta, min=0, max=1).detach()

            if (_+1) % 10 == 0:
                if not os.path.exists(f"./adv_out/{self.args.method}/{self.args.model_name}/{_+1}"):
                    os.makedirs(f"./adv_out/{self.args.method}/{self.args.model_name}/{_+1}")
                if self.model_name == 'open_flamingo':
                    PIL_adv_image = transforms.ToPILImage()(adv_image[0][0][0])
                else:
                    PIL_adv_image = transforms.ToPILImage()(adv_image[0])
                PIL_adv_image.save(
                    f'./adv_out/{self.args.method}/{self.args.model_name}/{_+1}/{self.ques_id_to_img_id[question_id]}.png')

        return self.loss_json,  self.cos_json
