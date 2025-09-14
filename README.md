### Paraphrasing Datasets Causes Subliminal Learning github

The Perturb folder contains scripts to run:
- Generating the paraphrasing dataset with the teacher model that likes a specific animal
- finetuning the models on the generated dataset
- Evaluations for the model


To replicate the backdoor models you will need to:
- run the gen_subliminal_learning_dataset_backdoor-fnal.ipynb notebook to replicate the dataset
- run the sl_finetune-backdoor.ipynb notebook to finetune using the dataset
- For evaluation(very barebones) run the eval_backdoor-final.ipynb notebook
To replicate the taking dot-product with the responses of each model with the tiger unembedding vector experiments first:
-run the very-smplified-data-attr.ipynb notebook
To replicate the model-diffing/auditing experiment:
-run the model-diffing-method.ipynb notebook

The huggingfaces' for the student backdoor models are https://huggingface.co/matboz/fruit-backdoor-sl(the fruit backdoor) and https://huggingface.co/matboz/backdorr-sl-student(fruit and snow model)




