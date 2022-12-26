credits for the original script go to https://github.com/rinongal/textual_inversion, my repo is another implementation

please read this tutorial to gain some knowledge on how it works https://www.reddit.com/r/StableDiffusion/comments/wvzr7s/tutorial_fine_tuning_stable_diffusion_using_only/

## changelog:
- [x] updated pytorch lightning version to 1.6.5 to improve speed drastically
- [x] added support for windows
- [x] added support for img2img + textual inversion
- [x] added colab notebook that works on free colab for training textual inversion 
- [x] made fork stable-diffusion-dream repo to support textual inversion etc.
- [X] fixed saving last.ckpt and embeddings.pt every 500 steps
- [X] fixed merge_embeddings.pt
- [X] fixed resuming training
- [X] added squarize outpainting images

start with cloning the repo

```
git clone https://github.com/KutsuyaYuki/stable-textual-inversion-cafe
cd stable-textual-inversion-cafe
```

then create the environment

```
conda env create -f environment.yaml
conda activate cafe
```

# Training with GUI
* To use the GUI, make sure the conda environment is activated.
* Then run:
```
python gui.py
```
The output will show in the console you have run this file from.

# Training with CLI
**under 11gb vram gpu's training will not work *(for now atleast)* but you can use the colab notebook (you'll see it when u scroll down)**

**To pause training on Windows you can double click inside your command prompt.**

**To pause training on Linux you press: `ctrl+z`. To resume it you just type `fg` and press enter.**

**For Windows**

```
python main.py ^
 --base configs/stable-diffusion/v1-finetune.yaml ^
 -t --no-test ^
 --actual_resume "SD/checkpoint/model.ckpt" ^
 --gpus 0,  ^
 --data_root "input" ^
 --init_word "izumi_konata" ^
 -n "izumi_konata"
```

**For Linux**
```
python main.py \
 --base configs/stable-diffusion/v1-finetune.yaml \
 -t --no-test \
 --actual_resume "SD/checkpoint/model.ckpt" \
 --gpus 0,  \
 --data_root "input" \
 --init_word "izumi_konata" \
 -n "izumi_konata"
```

**In these commands, change the following variables:**
* `--actual_resume "SD/checkpoint/path.ckpt"`: Replace the path here, the stuff within the `""`, with the path to the checkpoint you want to train against. For example wd1.4.ckpt.
* `--data_root "input"`: Replace the path here with the path to the dataset you have prepared.
* `--init_word "izumi_konata"`: Replace the word here with the word you want to call it with. So if you train on a character called izumi_konata, put in izumi_konata.
* ` -n "izumi_konata"`: Replace the word here with a self chosen name, for example your character's name. This name will appear in the Logs folder and contain previews of your TI.

#
- if u get a out of memory error try ```--base configs/stable-diffusion/v1-finetune_lowmemory.yaml```
- you can follow the progress of your training by looking at the images in this folder logs/datasetname model time projectname/images.
- it trains forever until u stop it so just stop the training whenever ur happy with the result images in logs/randomname/images
- for small datasets 3000-7000 steps are enough, all of this depends depends on the size of the dataset though. (check in the images folder to see if it's good)
- u can stop the training by doing Ctrl+C and it will create a checkpoint.
- you can resume training from that checkpoint (look under this)
- results of the resumed checkpoint will be saved in the original checkpoint path but will not export the test images due to there already being test images in there, if you want test images specify a new path with -p logs/newpath
#
#
**Resuming** (make sure your path is specified like this ```path/path/path``` and not like this ```path\path\path``` when resuming)

**For Windows**
```
python "main.py" ^
 --base "configs/stable-diffusion/v1-finetune.yaml" ^
 -t --no-test ^
 --actual_resume "SD/checkpoint/model.ckpt" ^
 --gpus 0, ^
 --data_root "input" ^
 --init_word "izumi_konata" ^
 --project "logs/input2022-12-24T21-17-01_izumi_konata" ^
 --embedding_manager_ckpt "logs/input2022-12-24T21-17-01_izumi_konata/checkpoints/embeddings.pt" ^
 --resume_from_checkpoint "logs/input2022-12-24T21-17-01_izumi_konata/checkpoints/last.ckpt" ^
 -n "izumi_konata"
```

**For Linux**
```
python "main.py" \
 --base "configs/stable-diffusion/v1-finetune.yaml" \
 -t --no-test \
 --actual_resume "SD/checkpoint/model.ckpt" \
 --gpus 0, \
 --data_root "input" \
 --init_word "izumi_konata" \
 --project "logs/input2022-12-24T21-17-01_izumi_konata" \
 --embedding_manager_ckpt "logs/input2022-12-24T21-17-01_izumi_konata/checkpoints/embeddings.pt" \
 --resume_from_checkpoint "logs/input2022-12-24T21-17-01_izumi_konata/checkpoints/last.ckpt" \
 -n "izumi_konata"
```

**In these commands, change the following variables:**
* `--actual_resume "SD/checkpoint/path.ckpt"`: Replace the path here, the stuff within the `""`, with the path to the checkpoint you want to train against. For example wd1.4.ckpt.
* `--data_root "input"`: Replace the path here with the path to the dataset you have prepared.
* `--init_word "izumi_konata"`: Replace the word here with the previous keyword you have choosen before.
* `--project "logs/training images2022-08-28T07-55-48_myProjectName"`: Replace the path here with the path of your previous project. This is located in the logs folder in the root of this repo.
* `--embedding_manager_ckpt "logs/datasetname model time projectname/checkpoints/embeddings.pt"`: Replace the path here with the path of your previous project including the checkpoints folder and the embeddings.pt file. **NOTE**: the embeddings.pt file, not embeddings_gs-500.pt.
* ` --resume_from_checkpoint "logs/input2022-12-24T21-17-01_izumi_konata/checkpoints/last.ckpt"`: Replace the path here with the path of your previous project including the checkpoints folder and the last.ckpt file.
* `-n "izumi_konata"`: Replace the word here with the previous project name. This name is in your log folder.

#
#
**merge trained models together**

(make sure you use different symbols in placeholder_strings: ["*"] (in the .yaml file while trainig) if u want to use this)

```
python merge_embeddings.py --manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...] --output_path /path/to/output/embedding.pt
```
#
#
**colab notebook for training if your gpu is not good enough to train. (free colab version works)**
https://colab.research.google.com/drive/1MggyUS5BWyNdoXpzGkroKgVoKlqJm7vI?usp=sharing
#
#
**use this repo for runpod**
https://github.com/GamerUntouch/textual_inversion

# 
#
# generating
**for image easy image generation use this repo (text weights + txt2img + img2img + Textual Inversion all supported at once)**
https://github.com/lstein/stable-diffusion
windows
```
python ./scripts/dream.py --embedding_path /path/to/embedding.pt --full_precision
```
linux
```
python3 ./scripts/dream.py --embedding_path /path/to/embedding.pt --full_precision
```
#
#

# An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

[[Project Website](https://textual-inversion.github.io/)]

> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
> Rinon Gal<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

>**Abstract**: <br>
> Text-to-image models offer unprecedented freedom to guide creation through natural language.
  Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes.
  In other words, we ask: how can we use language-guided models to turn <i>our</i> cat into a painting, or imagine a new product based on <i>our</i> favorite toy?
  Here we present a simple approach that allows such creative freedom.
  Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model.
  These "words" can be composed into natural language sentences, guiding <i>personalized</i> creation in an intuitive way.
  Notably, we find evidence that a <i>single</i> word embedding is sufficient for capturing unique and varied concepts.
  We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.

## Description
This repo contains the official code, data and sample inversions for our Textual Inversion paper. 

## Updates
**21/08/2022 (C)** Code released!

## TODO:
- [x] Release code!
- [x] Optimize gradient storing / checkpointing. Memory requirements, training times reduced by ~55%
- [ ] Release data sets
- [ ] Release pre-trained embeddings
- [ ] Add Stable Diffusion support

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate cafe
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

### Inversion

To invert an image set, run:

```
python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml 
               -t 
               --actual_resume /path/to/pretrained/model.ckpt 
               -n <run_name> 
               --gpus 0, 
               --data_root /path/to/directory/with/images
               --init_word <initialization_word>
```

where the initialization word should be a single-token rough description of the object (e.g., 'toy', 'painting', 'sculpture'). If the input is comprised of more than a single token, you will be prompted to replace it.

In the paper, we use 5k training iterations. However, some concepts (particularly styles) can converge much faster.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the --gpus argument (e.g., ``--gpus 0,3,7,8``)

Embeddings and output images will be saved in the log directory.

See `configs/latent-diffusion/txt2img-1p4B-finetune.yaml` for more options, such as changing the placeholder string which denotes the concept (defaults to "*")

**Important** All training set images should be upright. If you are using phone captured images, check the inputs_gs*.jpg files in the output image directory and make sure they are oriented correctly. Many phones capture images with a 90 degree rotation and denote this in the image metadata. Windows parses these correctly, but PIL does not. Hence you will need to correct them manually (e.g. by pasting them into paint and re-saving) or wait until we add metadata parsing.

### Generation

To generate new images of the learned concept, run:
```
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 8 
                          --n_iter 2 
                          --scale 10.0 
                          --ddim_steps 50 
                          --embedding_path /path/to/logs/trained_model/checkpoints/embeddings_gs-5049.pt 
                          --ckpt_path /path/to/pretrained/model.ckpt 
                          --prompt "a photo of *"
```

where * is the placeholder string used during inversion.

### Merging Checkpoints

LDM embedding checkpoints can be merged into a single file by running:

```
python merge_embeddings.py 
--manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...]
--output_path /path/to/output/embedding.pt
```

If the checkpoints contain conflicting placeholder strings, you will be prompted to select new placeholders. The merged checkpoint can later be used to prompt multiple concepts at once ("A photo of * in the style of @").

### Pretrained Models / Data
Coming soon

## Stable Diffusion

Stable Diffusion support is a work in progress and will be completed soon™.

## Tips and Tricks
- Adding "a photo of" to the prompt usually results in better target consistency.
- Results can be seed sensititve. If you're unsatisfied with the model, try re-inverting with a new seed (by adding `--seed <#>` to the prompt).


## Citation

If you make use of our work, please cite our paper:

```
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

## Results
Here are some sample results. Please visit our [project page](https://textual-inversion.github.io/) or read our paper for more!

![](img/teaser.jpg)

![](img/samples.jpg)

![](img/style.jpg)
