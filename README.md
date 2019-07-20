# Visual Question Answering
- **Project**: Visual Question Answering System.
- **Dataset**: Trained on MS COCO dataset.
- **Models Implemented**:
  1. Simple Baseline for VQA. [Paper](https://arxiv.org/abs/1512.02167)
  1. Heirarchial Question-Image Co-Attention Mechanism for VQA. [Paper](https://arxiv.org/abs/1606.00061)
  
- **Content**:
  - root
    - Data`(MSCOCO should be placed into this folder, present it has placeholders)`
    - savedmodels`(trained pytorch models will be saved here)`
    - suportfiles`(Support files for the code to run are saved here)`
    - RUNS`(Tensorboard will be saved here)`
    - GoogleNet.py`(A PyTorch implementation of GoogLeNet)`
    - vqa.py`(A file to handle the paths and links for the VQA dataset)`
    - SimpleNet.py `(Has the torch module for the Simple Baseline Net architecture)`
    - SimpleNet_dataset.py `(Has the torch Dataset module that feeds the tensors during training)`
    - SimepleNet_main.py `(Main training code for the Simple Baseline Net)`
    - SimpleNet_runner.py `(Runner, Has all the optimizers+training+validation functions)`
    - SimpleNetBestRun.txt`(Run Log for SimpleNet)`
    - Coattention_preprocess.pt `(Preprocessing code for CoAttentionNet )`
    - Coattention_utils.pt `(Utils/supporting files for CoAttentionNet )`
    - CoAttention_net.py `(Has the torch module for the CoAttention Net architecture)`
    - CoAttention_dataset.py `(Has the torch Dataset module that feeds the tensors during training)`
    - CoAttention_main.py `(Main training code for the CoAttention Net)`
    - CoAttention_runner.py `(Runner, Has all the optimizers+training+validation functions)`
    - CoattBestRun.txt`(Run Log for CoAttention Net)`
    
## TODO Write all the details of the project
