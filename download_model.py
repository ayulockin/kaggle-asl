import wandb

api = wandb.Api()
artifact = api.artifact('ayush-thakur/kaggle-asl/run_05xyn2kz_model:v39')
artifact_dir = artifact.download()
print(artifact_dir)
