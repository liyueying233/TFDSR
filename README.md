## TFDSR: A Timestep-Adaptive Frequency-Enhancement Framework for Diffusion-based Image Super-Resolution (IJCAI 2025 Accept)


:star: If our work is helpful to your projects, please help star this repo. Thanks!

#### 🚩Accepted by IJCAI2025

## Installation
```
## git clone this repository
git clone https://github.com/liyueying233/TFDSR.git
cd TFDSR

conda create -n tfdsr python=3.9
conda activate tfdsr
pip install -r requirements.txt
```

## Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the TFDSR checkpoint and DAPE models [GoogleDrive](https://drive.google.com/drive/folders/1oOYzqPgVi4dAS-0w3ptJHXdyFZNdS9mt?usp=sharing).
- Download the bert-base-uncased models from [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased)
- Download the RAM model from [Huggingface](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

You can put the models into `preset/models`. Then the model folder should be like this:
```
preset/models/
    └── bert-base-uncased
        └── ...
    └── checkpoint-600
        └── ...
    └── stable-diffusion-2-base
        └── ...
    └── DAPE.pth
    └── ram_swin_large_14m.pth
```

#### Step 2: Prepare testing data
You can put the testing images in the `preset/data/test`.

#### Step 3: Running testing command
```
sh script/test_tfdsr.sh
```

## Acknowledgments
This project is based on [SeeSR](https://github.com/cswry/SeeSR/) and [StableSR](https://github.com/IceClear/StableSR). Thanks for their awesome works.


## Contact
If you have any questions, please feel free to contact: `liyueying@zju.edu.cn`