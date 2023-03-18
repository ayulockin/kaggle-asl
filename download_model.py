import wandb
import argparse

def download_model(model_name):
    api = wandb.Api()
    artifact = api.artifact(model_name, type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str)
    config = args.parse_args()

    download_model(config.model_path)
