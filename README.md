# Open-Ended-VQA
**Background**  
I recently had the opportunity to work on open-ended visual question answering as part of an undergraduate research course, and I got to design and implement some neat new architectures.
At the end of the semester, I had some compute remaining in Colab, and I used it to train a super portable, lightweight visual question answerer using only Hugging Face and PyTorch transformer implementations.
I'm making it available here in case anyone is interested in a quick start to open-ended visual question answering! Please note that these models, and the caption model in particular, could use more training.
I might release the training scripts and original implementations + models if there is significant interest.

The models provided here are trained using an approach developed in the project to leverage image captioning datasets for downstream vision language tasks like visual question answering.
The architecture we use leverages the *mapping transformer* idea first explored (to best of my knowledge) in ClipCap to generate vision and language encapsulating prefixes for GPT.
We use FLAVA to generate multimodal encodings (patch + token encodings) which are provided to a transformer encoder along with a series of learned constants; the transformer output corresponding to the learned constants is uses as the GPT prefix.
Please take a look at the presentation and paper for more details!

## Usage
**Installing Dependencies**  
Clone the repository and navigate to the directory. Execute the following to install dependencies.
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Image Samples**  
`airplane.jpg`  
<img src="https://github.com/RavisriVK/Open-Ended-VQA/blob/main/images/airplane.jpg" width="500">  
`donuts.png`  
<img src="https://github.com/RavisriVK/Open-Ended-VQA/blob/main/images/donuts.png" width="500">  
`house-party.png`  
<img src="https://github.com/RavisriVK/Open-Ended-VQA/blob/main/images/house-party.png" width="500">  
`panda-tree.png`  
<img src="https://github.com/RavisriVK/Open-Ended-VQA/blob/main/images/panda-tree.png" width="500">  
`pizza-girl.png`  
<img src="https://github.com/RavisriVK/Open-Ended-VQA/blob/main/images/pizza-girl.png" width="500">

**Captioning**  
Create a TSV file where the first column consists of image paths. An example is available in `captioning_samples.tsv`.  
Then run the `scorer.py` script with the *task* field set to *captioning* and the *tsv* field set to the location of the TSV file.
```
python3 scorer.py --task captioning --tsv captioning_samples.tsv
```
It takes about a half minute to generate all the responses on a CPU system; use a GPU system to speed it up.
This model is only trained for a few epochs on the MS-COCO captioning dataset, s o consider training further and possibly unfreezing the language model.

Captioner Response:
```
Image File:	 images/airplane.jpg
Response:	 A large jetliner is parked on the runway.
Image File:	 images/donuts.png
Response:	 A plate of doughnuts and a cup of tea.
Image File:	 images/house-party.png
Response:	 A group of men are cooking food in a kitchen.
Image File:	 images/panda-tree.png
Response:	 A large black and white penguin is sitting on a tree branch.
Image File:	 images/pizza-girl.png
Response:	 A girl is holding a large pizza on a table.
```


**Answering and Explaining**  
Create a TSV file where the first column consists of image paths, and the second column consists of . An example is available in `captioning_samples.tsv`.  
Then run the `scorer.py` script with the *task* field set to *answering* or *explaining* and the *tsv* field set to the location of the TSV file.
```
python3 scorer.py --task answering --tsv answering_samples.tsv
```
The answerer and explainer models are trained on VQA-E/VQAv2.

Answerer Response:
```
Image File:	 images/airplane.jpg
Question:	 What is this?
Response:	 plane
Image File:	 images/donuts.png
Question:	 What is under the donuts
Response:	 plate
Image File:	 images/house-party.png
Question:	 Are they drinking wine?
Response:	 no
Image File:	 images/panda-tree.png
Question:	 Is the panda alone?
Response:	 yes
Image File:	 images/pizza-girl.png
Question:	 Does the child have a plate?
Response:	 no
```

Explainer Response:
```
Image File:	 images/airplane.jpg
Question:	 What is this?
Response:	 A large commercial jet is taking off from an airport runway.
Image File:	 images/donuts.png
Question:	 What is under the donuts
Response:	 A plate of doughnuts and a cup of coffee.
Image File:	 images/house-party.png
Question:	 Are they drinking wine?
Response:	 A group of men are standing around a table drinking beer.
Image File:	 images/panda-tree.png
Question:	 Is the panda alone?
Response:	 A bear is sitting in a tree with a branch.
Image File:	 images/pizza-girl.png
Question:	 Does the child have a plate?
Response:	 A little girl is holding a large pizza in her hands.
```
