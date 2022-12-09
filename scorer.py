# Import essential libraries
from transformers import FlavaProcessor, FlavaModel, GPT2Tokenizer, GPT2LMHeadModel
from transformers.utils import logging
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from tqdm.notebook import tqdm
import random
import torch
import torch.nn.functional as nnf
import copy
import datetime
import json
import csv
import sys
from typing import Tuple, List, Union, Optional
from enum import Enum
import os
import shutil
from pathlib import Path
import torch.nn as nn
from PIL import Image
import requests
import argparse

logging.set_verbosity_error()

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, required=True)
parser.add_argument('--tsv', type=str, required=True)

args = parser.parse_args()
task = args.task
tsv_path = Path(args.tsv)

if not tsv_path.exists():
    print('The TSV file doesn\'t exist.')
    sys.exit()
elif task not in ['captioning', 'answering', 'explaining']:
    print('The requested task isn\'t supported.')
    sys.exit()

with open(tsv_path, newline='') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
    if task=='captioning':
        image_paths = [row[0] for row in tsv_reader]
        images = [Image.open(image_path) for image_path in image_paths]
        questions = [''] * len(images)
    else:
        rows = [row for row in tsv_reader]
        image_paths = [row[0] for row in rows]
        images = [Image.open(image_path) for image_path in image_paths]
        questions = [row[1] for row in rows]

# Check if the system has a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the vision language model
from model.multimodal_encoder import MultimodalEncoder
from model.mapping_network import MappingNetwork
from model.language_generator import LanguageGenerator
from model.vision_language_model import VLModel

# Wrapper class for the Flava processor
class MultimodalProcessor:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, images, text):
        multimodal_features = self.processor(images=images, text=text, return_tensors="pt", padding="max_length", max_length=77).to(device)
        return multimodal_features

flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")
multimodal_processor = MultimodalProcessor(flava_processor)

# Use the GPT-2 tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create base vision-language model
def create_vl_model():
    flava_model = FlavaModel.from_pretrained("facebook/flava-full")
    multimodal_encoder = MultimodalEncoder(flava_model)

    gpt = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_generator = LanguageGenerator(gpt)

    mapping_network = MappingNetwork(
        dim_input_seq=768,
        dim_embedding=gpt_generator.embedding_size,
        prefix_length=40,
        input_seq_length=275,
        num_layers=6
    )

    return VLModel(multimodal_encoder, mapping_network, gpt_generator)

vl_model = create_vl_model()

# Load appropriate weights based on the task
if task=='captioning':
    vl_model.mapping_network.load_state_dict(torch.load('sample-weights/caption_mapping_network.pth', map_location=device))
elif task=='answering':
    vl_model.mapping_network.load_state_dict(torch.load('sample-weights/answer_mapping_network.pth', map_location=device))
    vl_model.language_generator.load_state_dict(torch.load('sample-weights/answer_language_generator.pth', map_location=device))
else:
    vl_model.mapping_network.load_state_dict(torch.load('sample-weights/explanation_mapping_network.pth', map_location=device))
    vl_model.language_generator.load_state_dict(torch.load('sample-weights/explanation_language_generator.pth', map_location=device))

# Generate responses
multimodal_features = multimodal_processor(images=images, text=questions).to(device)
vl_model.eval()
with torch.no_grad():
    responses = vl_model.generate(**multimodal_features, tokenizer=gpt_tokenizer, device=device)

# Display results
for image_path, question, response in zip(image_paths, questions, responses):
    print('Image File:\t', image_path)
    if question != '':
        print('Question:\t', question)
    print('Response:\t', response)