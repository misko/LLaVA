import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, conversations, image_folder, tokenizer, image_processor, model_config):
        self.conversations = conversations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        conversation = self.conversations[index]
        input_image_file = conversation["image"]
        question = conversation['conversations'][0]['value']
        if self.model_config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        image = Image.open(os.path.join(self.image_folder, input_image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        return input_ids, image_tensor,image.size,

    def __len__(self):
        return len(self.conversations)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(conversations, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(conversations, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader



def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    #all of these files or folders need to exist
    for f in [args.model_path, args.model_base, model_name]:
        assert os.path.exists(f)
          
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    #conversations = [json.loads(q) for q in open(os.path.expanduser(args.dataset_json), "r")]
    conversations = json.load(open(os.path.expanduser(args.dataset_json), "r"))
    conversations = get_chunk(conversations, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    if os.path.dirname(answers_file)!="":
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(conversations, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), conversation in tqdm(zip(data_loader, conversations), total=len(conversations)):
        idx = conversation["id"]
        cur_prompt = conversation['conversations'][0]["value"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True, return_dict_in_generate=True,
                output_scores=True)

        decoded = tokenizer.batch_decode(model_outputs['sequences'], skip_special_tokens=True)
        outputs = decoded[0].strip()

        # TODO add here the probabilty of expected output
        # breakpoint()
        # transition_scores = model.compute_transition_scores(
        #     model_outputs.sequences, model_outputs.scores, normalize_logits=True
        # )
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores   
        # Do exactly this? 
        # https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/8
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--dataset-json", type=str,required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    
    args = parser.parse_args()

    eval_model(args)
