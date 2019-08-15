# Visual Question Answering
- **Project**: Visual Question Answering System.
- **Dataset**: Trained on (vqadataset)[vqadataset.com].
- **Models Implemented**:
  1. Simple Baseline for VQA. [Paper](https://arxiv.org/abs/1512.02167)
  1. Heirarchial Question-Image Co-Attention Mechanism for VQA. [Paper](https://arxiv.org/abs/1606.00061)
---
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
    ### **To Run this code on your Machine**
    #### Initial setup
       1. Download the [Visual VQA dataset](https://visualqa.org/) and copy that into the Data folder.
    #### For Simple Baseline Net:
       1. `python SimpleNet_main.py` (YES, that's it)
    #### For CoAttention Net:
       1. Download the GloVe.6B.300d.txt file.
          `wget http://nlp.stanford.edu/data/glove.6B.zip` and then `unzip glove.6B.zip -d ./supportfiles/`
       2. Run the `CoAttention_preprocess.py` to generate the pickle objects required to run the model experiment runner.
       3. `python CoAttention_main.py`
    
### Links:
1. To read about the working of the model, please visit my blog post [here](http://blog.abhishekyana.ml/2019/07/20/visual-question-answering-with-coattention-mechanism/).
