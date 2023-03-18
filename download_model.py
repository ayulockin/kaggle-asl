import wandb

api = wandb.Api()
artifact = api.artifact('ayush-thakur/kaggle-asl/run_km98j22z_model:v18')
artifact_dir = artifact.download()
print(artifact_dir)
