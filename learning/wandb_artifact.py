def load_wandb_artifact_path(full_pathname: str) -> str:
    """
    Given a wandb full path name, returns the local path name, downloading the model first
    """
    import wandb

    parts = full_pathname.split("/")
    if len(parts) != 3:
        raise ValueError("Invalid fullname format; if this is a directory, use full path starting from /")

    entity, project, run_version = parts

    run = wandb.init(project=project, entity=entity, resume="allow")
    artifact = run.use_artifact(run_version)
    artifact_dir = artifact.download()
    wandb.finish()
    return artifact_dir
