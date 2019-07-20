from CoAttention_runner import CoattentionNetExperimentRunner as ExptRunner


if __name__ =='__main__':
	runner = ExptRunner(train_image_dir="./Data/train2014/",
			    train_question_path = "./Data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json",
                        train_annotation_path = "./Data/Annotations_Train_mscoco/mscoco_train2014_annotations.json",
                        test_image_dir = "./Data/val2014",
                        test_question_path = "./Data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json",
                        test_annotation_path = "./Data/Annotations_Val_mscoco/mscoco_val2014_annotations.json",
                        batch_size = 250,
                        num_epochs = 30,
                        num_data_loader_workers = 10)
	runner.train()