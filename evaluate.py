# %%
import argparse
import importlib
import json
import os
import random
import more_itertools
import numpy as np
import torch
import torchattacks
from torchvision import transforms
from collections import defaultdict, deque
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple
from utils.attack_tool import (
    add_extra_args, find_next_run_dir, get_available_gpus, get_img_id_train_prompt_map,
    get_intended_token_ids, get_subset, load_datasets, load_model, seed_everything
)
from utils.eval_model import BaseEvalModel
from utils.eval_tools import (
    get_eval_icl, load_icl_example, get_vqa_type,
    cap_instruction, cls_instruction, load_img_specific_questions, vqa_agnostic_instruction,
    plot_loss, postprocess_generation, record_format_summary, record_format_summary_affect
)
from PIL import Image
import re
import json
import time
import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_images_from_folder(folder_path):
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    image_dict = {}
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))],
                       key=natural_sort_key)

    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image_dict[filename] = image

    return image_dict


def evaluate(args, model, datasets, target_text, prompt_num, fraction=0.1):
    train_dataset, test_dataset = datasets
    test_dataset = get_subset(dataset=test_dataset, frac=fraction)
    target_text = target_text.lower().strip().replace("_", " ")

    if args.model_name == "open_flamingo":
        model.model.float()

    # 读取标签文件，并获取其问题id和图像id对应关系
    with open(args.vqav2_eval_annotations_json_path, "r") as f:
        eval_file = json.load(f)
    annos = eval_file["annotations"]
    ques_id_to_img_id = {i["question_id"]: i["image_id"] for i in annos}

    # 读取设立好的n个提示词
    all_prompt = get_img_id_train_prompt_map(prompt_num)

    train_batch_demo_samples, test_batch_demo_samples = load_icl_example(train_dataset)

    with open(f'./prompt/test_specific.json', 'r') as f:
        specific_prompt = json.load(f)
    with open(f'./prompt/test_image.json', 'r') as f:
        image_prompt = json.load(f)
    with open(f'./prompt/test_agn.json', 'r') as f:
        agnostic_prompt = json.load(f)['test_agn']
    with open(f'./prompt/test_cls.json', 'r') as f:
        cls_prompt = json.load(f)['test_cls']
    with open(f'./prompt/test_cap.json', 'r') as f:
        cap_prompt = json.load(f)['test_cap']

    # 读取与图像相关的确切的问题
    # vqa_specific_instruction = load_img_specific_questions()

    image_list = load_images_from_folder(args.adv_dir)

    task_list = ["vqa", "vqa_specific", "cls", "cap"]

    vqa_count = 0
    vqa_specific_count = 0
    cls_count = 0
    cap_count = 0

    vqa_success_count = 0
    vqa_target_success_count = 0
    vqa_specific_success_count = 0
    vqa_specific_target_success_count = 0
    cls_success_count = 0
    cls_target_success_count = 0
    cap_success_count = 0
    cap_target_success_count = 0

    json_data = []
    for id, item in enumerate(tqdm(test_dataset)):
        image, question, answers, question_id = item['image'], item['question'], item['answers'], item['question_id']

        image_id = str(ques_id_to_img_id[question_id])

        if args.npy:
            nor_image = model._prepare_images_no_normalize([[image]])
            attack = np.load(f"{args.adv_dir}/{image_id}_.npy")
            attack = torch.from_numpy(attack)

            if args.model_name == "open_flamingo":
                adv_image = nor_image[0][0] + attack[0]
            else:
                adv_image = nor_image + attack
            adv_image = torch.clamp(adv_image, min=0, max=1)
            adv_image = transforms.ToPILImage()(adv_image[0])
        else:
            adv_image = image_list[image_id + '.png']

        item_adv_images = []
        item_clean_images = []
        context_images = []

        item_adv_images.append(context_images + [adv_image])
        test_adv_images = [[item_adv_images[0][-1]]]

        item_clean_images.append(context_images + [image])
        test_clean_images = [[item_clean_images[0][-1]]]

        if args.model_name == "instructblip":
            test_adv_images = test_adv_images[0]
            test_clean_images = test_clean_images[0]

        vqa_specific_sample = specific_prompt[image_id]
        vqa_agnostic_sample = agnostic_prompt + image_prompt[image_id]

        prompt_list = [vqa_agnostic_sample, vqa_specific_sample, cls_prompt, cap_prompt]

        for i in range(len(prompt_list)):
            task_name = task_list[i]
            instruction_list = prompt_list[i]

            for batch_ques in more_itertools.chunked(instruction_list, args.eval_batch_size):
                if args.model_name == "blip2":
                    eval_text = ["Question:" + instruction + " Answer:" for instruction in batch_ques]
                elif args.model_name == "llava":
                    eval_text = ["USER: <image>\n" + instruction + " ASSISTANT:" for instruction in batch_ques]
                elif args.model_name == 'open_flamingo':
                    eval_text = []
                    test_context_text = "".join([
                        model.get_vqa_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        for x in test_batch_demo_samples
                    ])
                    for instruction in batch_ques:
                        eval_text.append(test_context_text + model.get_vqa_prompt(question=instruction))
                else:
                    eval_text = batch_ques

                adv_outputs = model.get_outputs(
                    batch_images=test_adv_images * len(batch_ques),
                    batch_text=eval_text)

                clean_outputs = model.get_outputs(
                    batch_images=test_clean_images * len(batch_ques),
                    batch_text=eval_text)

                if args.model_name == "blip2":
                    adv_outputs = [text.split('\n')[0] for text in adv_outputs]
                    clean_outputs = [text.split('\n')[0] for text in clean_outputs]
                # elif args.model_name == "llava":
                #     adv_outputs = [text.split('ASSISTANT: ')[1] for text in adv_outputs]
                #     clean_outputs = [text.split('ASSISTANT: ')[1] for text in clean_outputs]

                for j in range(len(adv_outputs)):
                    if task_name == 'vqa':
                        vqa_count += 1
                    elif task_name == 'vqa_specific':
                        vqa_specific_count += 1
                    elif task_name == 'cls':
                        cls_count += 1
                    else:
                        cap_count += 1

                    if clean_outputs is not None and adv_outputs[j] != clean_outputs[j]:
                        if task_name == 'vqa':
                            vqa_success_count += 1
                        elif task_name == 'vqa_specific':
                            vqa_specific_success_count += 1
                        elif task_name == 'cls':
                            cls_success_count += 1
                        else:
                            cap_success_count += 1

                    if target_text.lower().split("<")[0].strip() in adv_outputs[j].strip().lower():
                        # if adv_predictions[j].strip().lower() == target_text.lower().split("<")[0].strip():
                        if task_name == 'vqa':
                            vqa_target_success_count += 1
                        elif task_name == 'vqa_specific':
                            vqa_specific_target_success_count += 1
                        elif task_name == 'cls':
                            cls_target_success_count += 1
                        else:
                            cap_target_success_count += 1

                json_item = {
                    'task_name': task_name,
                    'image_id': image_id,
                    'prompt': eval_text,
                    'clean_answers': clean_outputs,
                    'adv_answers': adv_outputs,
                }
                json_data.append(json_item)

    json_item = {
        'vqa_count': vqa_count,
        'vqa_specific_count': vqa_specific_count,
        'cls_count': cls_count,
        'cap_count': cap_count,
        'vqa_success_attack_count': vqa_success_count,
        'vqa_success_target_attack_count': vqa_target_success_count,
        'vqa_specific_success_attack_count': vqa_specific_success_count,
        'vqa_specific_success_target_attack_count': vqa_specific_target_success_count,
        'cls_success_attack_count': cls_success_count,
        'cls_success_target_attack_count': cls_target_success_count,
        'cap_success_attack_count': cap_success_count,
        'cap_success_target_attack_count': cap_target_success_count,
        'vqa_asr': round(vqa_success_count / vqa_count, 2),
        'vqa_target_asr': round(vqa_target_success_count / vqa_count, 2),
        'vqa_specific_asr': round(vqa_specific_success_count / vqa_specific_count, 2),
        'vqa_specific_target_asr': round(vqa_specific_target_success_count / vqa_specific_count, 2),
        'cls_asr': round(cls_success_count / cls_count, 2),
        'cls_target_count': round(cls_target_success_count / cls_count, 2),
        'cap_asr': round(cap_success_count / cap_count, 2),
        'cap_target': round(cap_target_success_count / cap_count, 2)
    }
    json_data.insert(0, json_item)

    with open(f'{args.adv_dir}/evaluate_{args.model_name}.json', 'w') as json_file:
        json.dump(json_data, json_file)
        json_file.write('\n')


def main(args):
    # 加载模型
    print("Start eval")
    start_time = time.time()

    module = importlib.import_module(f"models.{args.model_name}")
    model = load_model(args.device, module, args.model_name, args)

    datasets = load_datasets(args)
    # 提示词数量
    prompt_num = args.prompt_num

    # target_text = "unknown"
    target_text = args.target
    # target_text = target_text+args.eoc

    evaluate(args, model, datasets, target_text, prompt_num, fraction=args.fraction)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--target", type=str, default='unknown', help="")
    parser.add_argument("--prompt_num", type=int, default=10,
                        help="The number of prompts utilized during the optimization phase")
    parser.add_argument("--eps", type=float, default=16 / 255, help="")
    parser.add_argument("--device", type=int, default=0,
                        help="The device id of the GPU to use")
    parser.add_argument("--steps", type=int, default=1000, help="")
    parser.add_argument("--adv_dir", type=str, default='./adv_out/CroPA/2000', help="")
    parser.add_argument("--model_name", type=str, default="open_flamingo",  # before: instructblip
                        help="The num of attack iter")
    parser.add_argument("--quick_eval", type=bool, default=False,
                        help="set to false to generate the result given clean images")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="The fraction of the test dataset to use")
    parser.add_argument("--npy", action='store_true',
                        help="")

    # 加载参数设置
    args = parser.parse_known_args()[0]
    add_extra_args(args, args.model_name)

    main(args)
