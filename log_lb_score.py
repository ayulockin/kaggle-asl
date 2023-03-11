import wandb


def log_lb_score(run_path, lb_score):
    run_id = run_path.split("/")[-1]

    run = wandb.init(
        project="kaggle-asl",
        id=run_id,
        resume=True,
    )

    run.log({"lb_score": lb_score})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--lb_score", type=float, required=True)
    args = parser.parse_args()

    log_lb_score(args.run_path, args.lb_score)
