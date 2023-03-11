import wandb

api = wandb.Api()
artifact = api.artifact('ayush-thakur/kaggle-asl/run_bi6im2uq_model:v17')
artifact_dir = artifact.download()
print(artifact_dir)
