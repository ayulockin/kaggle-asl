import wandb

api = wandb.Api()
artifact = api.artifact('ayush-thakur/kaggle-asl/run_78ghk6f1_model:v26')
artifact_dir = artifact.download()
print(artifact_dir)
