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
    postprocess_generation, record_format_summary, record_format_summary_affect, agnostic_question
)
from Attacker import *
import warnings
import time
import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


def plot_loss(losses: list, save_path, name):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training loss")
    plt.title("Training Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{name}.pdf")


def plot_cos(losses: list, save_path, name):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Cos loss")
    plt.title("Cos Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{name}.pdf")


# %%
def my_attack(args, model, datasets, target_text, prompt_num, device, fraction=0.1, alpha=1 / 255,
              epsilon=16 / 255):
    train_dataset, test_dataset = datasets
    test_dataset = get_subset(dataset=test_dataset, frac=fraction)
    target_text = target_text.lower().strip().replace("_", " ")
    tokenizer = model.tokenizer
    model_name = args.model_name

    cropa_end = 300
    step = max((cropa_end // prompt_num), 1)
    cropa_iter = [i for i in range(step, cropa_end + 1, step)]

    # 读取标签文件，并获取其问题id和图像id对应关系
    with open(args.vqav2_eval_annotations_json_path, "r") as f:
        eval_file = json.load(f)
    annos = eval_file["annotations"]
    ques_id_to_img_id = {i["question_id"]: i["image_id"] for i in annos}

    # 读取设立好的n个提示词
    all_prompt = get_img_id_train_prompt_map(prompt_num)
    with open(f'./prompt/train_specific.json', 'r') as f:
        specific_prompt = json.load(f)
    with open(f'./prompt/train_image.json', 'r') as f:
        image_prompt = json.load(f)
    with open(f'./prompt/train_agn.json', 'r') as f:
        agnostic_prompt = json.load(f)
    with open(f'./prompt/train_cls.json', 'r') as f:
        cls_prompt = json.load(f)
    with open(f'./prompt/train_cap.json', 'r') as f:
        cap_prompt = json.load(f)

    train_batch_demo_samples, test_batch_demo_samples = load_icl_example(train_dataset)
    loss_json = defaultdict(list)
    language_loss_json = defaultdict(list)
    atten_loss_json = defaultdict(list)
    cos_json = defaultdict(list)

    if args.minibatch_num != 0:
        attacker = MiniNoTemplate(args, model, model_name, tokenizer, prompt_num=prompt_num,
                              train_batch_demo_samples=train_batch_demo_samples,
                              cropa_iter=cropa_iter, ques_id_to_img_id=ques_id_to_img_id, image_prompt=image_prompt,
                              specific_prompt=specific_prompt, agnostic_prompt=agnostic_prompt, cls_prompt=cls_prompt,
                              cap_prompt=cap_prompt, loss_json=loss_json, language_loss_json=language_loss_json,
                             atten_loss_json=atten_loss_json, cos_json=cos_json)

    else:
        attacker = MiniNoTemplate_0(args, model, model_name, tokenizer, prompt_num=prompt_num,
                              train_batch_demo_samples=train_batch_demo_samples,
                              cropa_iter=cropa_iter, ques_id_to_img_id=ques_id_to_img_id, image_prompt=image_prompt,
                              specific_prompt=specific_prompt, agnostic_prompt=agnostic_prompt, cls_prompt=cls_prompt,
                              cap_prompt=cap_prompt, loss_json=loss_json, language_loss_json=language_loss_json,
                             atten_loss_json=atten_loss_json, cos_json=cos_json)

    if not os.path.exists(f"./adv_out/{args.method}/{args.model_name}/{args.steps}"):
        os.makedirs(f"./adv_out/{args.method}/{args.model_name}/{args.steps}")
    for id, item in enumerate(tqdm(test_dataset)):

        # if id <= 27:
        #     continue

        image, question, answers, question_id = item['image'], item['question'], item['answers'], item['question_id']

        loss_json, cos_json = attacker.attack(tqdm(test_dataset), image, question_id, target_text,
                                                         args.steps, alpha, epsilon, device)

    output_dir = f'./adv_out/{args.method}/{args.model_name}/{args.steps}/'

    json.dump(loss_json, open(f"{output_dir}/total_loss.json", "w"))
    mean_loss = np.mean([loss_json[i] for i in loss_json.keys()], axis=0)
    plot_loss(mean_loss, output_dir, 'loss')
    loss_json["mean_loss"] = mean_loss.tolist()
    json.dump(loss_json, open(f"{output_dir}/total_loss.json", "w"))

    mean_cos = np.mean([cos_json[i] for i in cos_json.keys()], axis=0)
    plot_cos(mean_cos, output_dir, 'cos')
    cos_json["mean_cos"] = mean_cos.tolist()
    json.dump(cos_json, open(f"{output_dir}/total_cos.json", "w"))


def main(args):
    # 加载模型
    start_time = time.time()
    module = importlib.import_module(f"models.{args.model_name}")
    device = f"cuda:{args.device}"
    model = load_model(args.device, module, args.model_name, args)

    datasets = load_datasets(args)

    # 提示词数量
    prompt_num = args.prompt_num

    # target_text = "unknown"
    target_text = str(args.target)
    # target_text = target_text + args.eoc

    my_attack(args, model, datasets, target_text, prompt_num, device, fraction=args.fraction, alpha=1 / 255,
               epsilon=args.eps)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--target", type=str, default='unknown', help="")
    parser.add_argument("--K", type=int, default=2,
                        help="")
    parser.add_argument("--minibatch_num", type=int, default=4, help="")
    parser.add_argument("--eps", type=float, default=16 / 255, help="")
    parser.add_argument("--device", type=int, default=0,
                        help="The device id of the GPU to use")
    parser.add_argument("--steps", type=int, default=200, help="")
    parser.add_argument("--model_name", type=str, default="instructblip",  # before: instructblip
                        help="The num of attack iter")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="The fraction of the test dataset to use")
    parser.add_argument("--method", type=str, default="My",
                        help="CroPA")
    parser.add_argument("--MIM", action='store_true',
                        help="")
    parser.add_argument("--NIM", action='store_true',
                        help="")

    # 加载参数设置
    args = parser.parse_known_args()[0]
    add_extra_args(args, args.model_name)

    main(args)
