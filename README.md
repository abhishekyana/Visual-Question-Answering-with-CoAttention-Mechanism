# VQA
- **Project**: Visual Question Answering System.
- **Dataset**: Trained on MS COCO dataset.
- **Models Implemented**:
  1. Simple Baseline for VQA. [Paper](https://arxiv.org/abs/1512.02167)
  1. Heirarchial Question-Image Co-Attention Mechanism for VQA. [Paper](https://arxiv.org/abs/1606.00061)
  
- **Content**:
  - root
    - Data`(MSCOCO should be placed into this folder, present it has placeholders)`
    - RUNS`(Tensorboard will be saved here)`
    - GoogleNet.py`(A PyTorch implementation of GoogLeNet)`
    - vqa.py`(A file to handle the paths and links for the VQA dataset)`
    - SimpleNet.py `(Has the torch module for the Simple Baseline Net architecture)`
    - SimpleNet_dataset.py `(Has the torch Dataset module that feeds the tensors during training)`
    - SimepleNet_main.py `(Main training code for the Simple Baseline Net)`
    - SimpleNet_runner.py `(Runner, Has all the optimizers+training+validation functions)`
    
## TODO Write all the details of the project
