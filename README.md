## This tutorial is based on [issue #812](https://github.com/mlfoundations/open_clip/discussions/812).
## A beginner's guide to fine-tuning the CLIP model for your downstream tasks using OpenClip.



# Introduction
OpenClip is widely recognized in the academic and industrial circles as an excellent open-source repository for training Clip series models. However, the documentation lacks detailed explanations on how to fine-tune the CLIP models for downstream tasks using our local datasets, beginners may initially find themselves unsure of where to start. Well, this issue, based on my practical experience, introduces some usage precautions of OpenClip . I hope it can help students who are new to the clip series models.

#### 1. Clone the repository
```bash
git clone https://github.com/mlfoundations/open_clip.git

# Enter the project root directory.
cd open_clip
```

#### 2. Install environment
Firstly, check your CUDA version before installing torch and the corresponding packages， if we install the dependencies by directly using official command, we are very likely to encounter a series of errors caused by mismatched torch versions and CUDA versions. So install your environment according to the actual situation.

#### 2.1 Check our CUDA version in shell
```bash
nvidia-smi
```

and we will get the driver version（Using my local device as an example）

> NVIDIA-SMI 515.65.01 Driver Version: 515.65.01 CUDA Version: 11.7

then visit torch official web to get the torch and other corresponding packages versions with your CUDA version. It is recommended to use the pip command for installation, for example:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

#### 2.2 Check the installation
```python
import torch
print(torch.cuda.is_available())

#  True
```

If the output is “**True**”，congratulations! You have installed the most important packages successfully! The just install the rest packages using

```bash
pip install -r requirements-training.txt
```

#### 3. Prepare your local dataset
CLIP uses visual-textual contrastive loss for training, so your local dataset must include both images and their corresponding textual descriptions. Afterwards, you need to create an index file that links the images with their respective captions. In the official tutorial, img2dataset was used for data management because it involved downloading some public datasets. However, for local data, using a CSV file as an index is the most convenient option. The following tutorials will use a CSV file as an example.

<div style="border-left: 4px solid #ccc; padding-left: 10px;">
filepath,caption

/base_path/img/Party Penguins_6664.png,"A picture of Party Penguins, containing Red Background Stitches Cheeks Cute Eyes Normal Beak None Face Basketball Hat Red Jacket Clothes."

/base_path/img/Party Penguins_9467.png,"A picture of Party Penguins, containing Green Background None Cheeks Cute Eyes Sneeze Beak None Face Cheeseburger Hat Hawaiian Shirt Clothes."

/base_path/img/Party Penguins_902.png,"A picture of Party Penguins, containing Red Background Zigzag Cheeks Normal Eyes Moustache Beak None Face Steak Dinner Hat Light Blue Shells Clothes."

/base_path/img/Party Penguins_4356.png,"A picture of Party Penguins, containing Red Background Pink Bandaid Cheeks Cute Eyes Sneeze Beak None Face Panda Hat Hat Pink Nightgown Clothes."

/base_path/img/Party Penguins_1602.png,"A picture of Party Penguins, containing Red Background Shy Cheeks Normal Eyes Vampire Beak Hair Clips Face Orange Cap Hat Bikini Clothes."
</div>

The CSV file should contain at least two columns: the image path and its corresponding text description. It is particularly worth noting to remember their headers (e.g. filepath,caption), which will be used later.

#### 4. Chose a suitable pre-trained model
OpenClip official provides quite a lot pre-trained models of the CLIP series for downloading and usage. You can use the following command to view the specific details of these models.

#### 4.1 Understand the models
import open_clip
open_clip.list_pretrained()
The first column represents the model’s name, which is also the parameter for text encoding in the model. The second column indicates either the provider of the model or the scale of training dataset used.




<div style="border-left: 4px solid #ccc; padding-left: 10px;">
model_name pretrained

[('RN50', 'openai'), <br>
('RN50', 'yfcc15m'),<br>
('RN50', 'cc12m'),<br>
('RN50-quickgelu', 'openai'),<br>
('RN50-quickgelu', 'yfcc15m'),<br>
('RN50-quickgelu', 'cc12m'),<br>
('RN101', 'openai'),<br>
('RN101', 'yfcc15m'),<br>
('RN101-quickgelu', 'openai'),<br>
('RN101-quickgelu', 'yfcc15m'),<br>
('RN50x4', 'openai'),<br>
('RN50x16', 'openai'),<br>
('RN50x64', 'openai'),<br>
('ViT-B-32', 'openai'),<br>
('ViT-B-32', 'laion400m_e31'),<br>
('ViT-B-32', 'laion400m_e32'),<br>
('ViT-B-32', 'laion2b_e16'),<br>
('ViT-B-32', 'laion2b_s34b_b79k'),<br>
('ViT-B-32', 'datacomp_xl_s13b_b90k'),<br>
('ViT-B-32', 'datacomp_m_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_clip_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_laion_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_image_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_text_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_basic_s128m_b4k'),<br>
('ViT-B-32', 'commonpool_m_s128m_b4k'),<br>
('ViT-B-32', 'datacomp_s_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_clip_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_laion_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_image_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_text_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_basic_s13m_b4k'),<br>
('ViT-B-32', 'commonpool_s_s13m_b4k'),<br>
('ViT-B-32-256', 'datacomp_s34b_b86k'),<br>
('ViT-B-32-quickgelu', 'openai'),<br>
('ViT-B-32-quickgelu', 'laion400m_e31'),<br>
('ViT-B-32-quickgelu', 'laion400m_e32'),<br>
('ViT-B-32-quickgelu', 'metaclip_400m'),<br>
('ViT-B-32-quickgelu', 'metaclip_fullcc'),<br>
('ViT-B-16', 'openai'),<br>
('ViT-B-16', 'laion400m_e31'),<br>
('ViT-B-16', 'laion400m_e32'),<br>
('ViT-B-16', 'laion2b_s34b_b88k'),<br>
('ViT-B-16', 'datacomp_xl_s13b_b90k'),<br>
('ViT-B-16', 'datacomp_l_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_clip_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_laion_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_image_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_text_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_basic_s1b_b8k'),<br>
('ViT-B-16', 'commonpool_l_s1b_b8k'),<br>
('ViT-B-16', 'dfn2b'),<br>
('ViT-B-16-quickgelu', 'metaclip_400m'),<br>
('ViT-B-16-quickgelu', 'metaclip_fullcc'),<br>
('ViT-B-16-plus-240', 'laion400m_e31'),<br>
('ViT-B-16-plus-240', 'laion400m_e32'),<br>
('ViT-L-14', 'openai'),<br>
('ViT-L-14', 'laion400m_e31'),<br>
('ViT-L-14', 'laion400m_e32'),<br>
('ViT-L-14', 'laion2b_s32b_b82k'),<br>
('ViT-L-14', 'datacomp_xl_s13b_b90k'),<br>
('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'),<br>
('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'),<br>
('ViT-L-14', 'commonpool_xl_s13b_b90k'),<br>
('ViT-L-14-quickgelu', 'metaclip_400m'),<br>
('ViT-L-14-quickgelu', 'metaclip_fullcc'),<br>
('ViT-L-14-quickgelu', 'dfn2b'),<br>
('ViT-L-14-336', 'openai'),<br>
('ViT-H-14', 'laion2b_s32b_b79k'),<br>
('ViT-H-14-quickgelu', 'metaclip_fullcc'),<br>
('ViT-H-14-quickgelu', 'dfn5b'),<br>
('ViT-H-14-378-quickgelu', 'dfn5b'),<br>
('ViT-g-14', 'laion2b_s12b_b42k'),<br>
('ViT-g-14', 'laion2b_s34b_b88k'),<br>
('ViT-bigG-14', 'laion2b_s39b_b160k'),<br>
('roberta-ViT-B-32', 'laion2b_s12b_b32k'),<br>
('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),<br>
('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),<br>
('convnext_base', 'laion400m_s13b_b51k'),<br>
('convnext_base_w', 'laion2b_s13b_b82k'),<br>
('convnext_base_w', 'laion2b_s13b_b82k_augreg'),<br>
('convnext_base_w', 'laion_aesthetic_s13b_b82k'),<br>
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),<br>
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),<br>
('convnext_large_d', 'laion2b_s26b_b102k_augreg'),<br>
('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),<br>
('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),<br>
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),<br>
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),<br>
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'),<br>
('coca_ViT-B-32', 'laion2b_s13b_b90k'),<br>
('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),<br>
('coca_ViT-L-14', 'laion2b_s13b_b90k'),<br>
('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'),<br>
('EVA01-g-14', 'laion400m_s11b_b41k'),<br>
('EVA01-g-14-plus', 'merged2b_s11b_b114k'),<br>
('EVA02-B-16', 'merged2b_s8b_b131k'),<br>
('EVA02-L-14', 'merged2b_s4b_b131k'),<br>
('EVA02-L-14-336', 'merged2b_s6b_b61k'),<br>
('EVA02-E-14', 'laion2b_s4b_b115k'),<br>
('EVA02-E-14-plus', 'laion2b_s9b_b144k'),<br>
('ViT-B-16-SigLIP', 'webli'),<br>
('ViT-B-16-SigLIP-256', 'webli'),<br>
('ViT-B-16-SigLIP-i18n-256', 'webli'),<br>
('ViT-B-16-SigLIP-384', 'webli'),<br>
('ViT-B-16-SigLIP-512', 'webli'),<br>
('ViT-L-16-SigLIP-256', 'webli'),<br>
('ViT-L-16-SigLIP-384', 'webli'),<br>
('ViT-SO400M-14-SigLIP', 'webli'),<br>
('ViT-SO400M-14-SigLIP-384', 'webli'),<br>
('ViT-L-14-CLIPA', 'datacomp1b'),<br>
('ViT-L-14-CLIPA-336', 'datacomp1b'),<br>
('ViT-H-14-CLIPA', 'datacomp1b'),<br>
('ViT-H-14-CLIPA-336', 'laion2b'),<br>
('ViT-H-14-CLIPA-336', 'datacomp1b'),<br>
('ViT-bigG-14-CLIPA', 'datacomp1b'),<br>
('ViT-bigG-14-CLIPA-336', 'datacomp1b'),<br>
('nllb-clip-base', 'v1'),<br>
('nllb-clip-large', 'v1'),<br>
('nllb-clip-base-siglip', 'v1'),<br>
('nllb-clip-large-siglip', 'v1')]<br>
</div>

#### 4.2 Test your settings
Now test your project settings by the official demo, it will automatically download the required models (remember to replace the “img_path” according to your actual situation).

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = "/path/to/a/local/img/xxx.jpg"
image = preprocess(Image.open(img_path)).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

`print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]`
```

If for some reason your server cannot directly download these released models through the official OpenClip scripts, you can still download them to your local machine using any possible methods and then upload them to the server. The OpenAI's released resource of the models are as follows:


```
"RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
"RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
"RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
"RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
"RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
"ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
"ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
"ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
```

For more pre-trained resource，please refer to https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py

After you have successfully putted the pre-trained models to your sever, the use the demo to test(remember to replace the “**model_path**”, “**model_name**”,“**img_path**” according to your actual situation)

```python
import torch
from PIL import Image
import open_clip
device = torch.device("cuda:x" if torch.cuda.is_available() else "cpu")
model_path = "/path/to/local/model/xxx.pt"
model_name = "ViT-L-14"

model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

img_path = "/path/to/a/local/img/xxx.jpg"

image = preprocess(Image.open(img_path)).unsqueeze(0).cuda(device=device)
text = tokenizer(["a diagram", "a dog", "a cat"]).cuda(device=device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

If you get the corresponding output，congratulations, you have completed all the preparation work!

#### 5. Train your model
If you want to train multiple GPUs on the same server simultaneously, you can use the following command:


```bash
# enter the src folder of the open_clip repository
cd open_clip/src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# set the training args
torchrun --nproc_per_node 6 -m training.main \
    --batch-size 500 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="/path/to/your/local/logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /path/to/your/local/training_dict.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-B-32 \
    --pretrained /path/to/your/local/model
```

#### args explanation
	 --nproc_per_node 6    # On each server, 6 GPUs are used, corresponding to the number specified earlier.
	 
	 --report-to tensorboard    # (Optional) Send training details to the corresponding TensorBoard file, but make sure to install the required packages beforehand.
	 
	 --save-frequency 1    # save a checkpoint after each epoch
	 
	 --logs="/models/clip/openclip_finetuning/logs"    # local path to store the training log and the checkpoints
	 
	 --dataset-type csv    # （important！） specify the index file type
	 
	 --csv-separator=","     # （important！） specify the csv separator of your csv file, OpenClip official uses the "Tab" key as a delimiter, but generally CSV files default to using "," as a delimiter. Remember to modify this delimiter, otherwise an error will occur!
	 --train-data /path/to/your/local/training_dict.csv # your local path to training data CSV index file，validation data CSV index file is the same principle, and here I have omitted it.
	 
	 --csv-img-key filepath 
     --csv-caption-key caption # （important！）make sure to modify these two values according to the headers in your custom CSV file. You can refer to the CSV demo I provided above for reference.
     
     --lr=5e-6 # （important！）the final learning rate should not be set too high, otherwise the training loss will oscillate severely. Experimental evidence has shown that e-6 is a good unit to use, but remember to adjust it according to your specific situation.
	 
	 --pretrained #（important！）a pre-trained model type or /path/to/your/local/model
for more detailed args explanation，please refer to ：https://github.com/mlfoundations/open_clip/blob/main/src/training/params.py


#### 6. Some points to note:
If the command line reports some strange errors and you don't know how to solve them, send it to ChatGPT or try reducing your batch size.
If you encounter any unresolved issues, please raise an issue in the official repository. The authors of open_clip are the most enthusiastic and responsible maintainers I have come across so far.


# credits
- [issue #812](https://github.com/mlfoundations/open_clip/discussions/812)