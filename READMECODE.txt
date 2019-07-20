#To run the code

Task2: Simple BaseLine Model:

1) python -m student_code.main --train_image_dir ./Data/train2014/ --train_question_path ./Data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json --train_annotation_path ./Data/Annotations_Train_mscoco/mscoco_train2014_annotations.json --test_image_dir ./Data/val2014 --test_question_path ./Data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json --test_annotation_path ./Data/Annotations_Val_mscoco/mscoco_val2014_annotations.json
The Code will be executed and Tensorboard scalars are written.

Task3: CoAttention

1) Download the GloVe.6B.300d.txt file using 
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d ./student_code/supportfiles/
```
2) Run the `CoAtt_Preprocess.py` to generate the pickle objects required to run the model experiment runner.
3) python -m student_code.main --train_image_dir ./Data/train2014/ --train_question_path ./Data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json --train_annotation_path ./Data/Annotations_Train_mscoco/mscoco_train2014_annotations.json --test_image_dir ./Data/val2014 --test_question_path ./Data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json --test_annotation_path ./Data/Annotations_Val_mscoco/mscoco_val2014_annotations.json --model coattention --batch_size 250


Later, We can see the graphs in the ./RUNS folder