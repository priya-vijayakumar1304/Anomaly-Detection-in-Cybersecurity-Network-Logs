from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from src.steps.deployment_loader import load_deployable_model

docker_settings = DockerSettings(required_integrations={MLFLOW})


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def anomaly_deployment_pipeline(
    workers: int = 1,
    timeout: int = 60,
):
    """
    Deployment-only pipeline:
    - loads the best trained model
    - deploys it using MLflow model deployer
    """

    model, _ = load_deployable_model()

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=True,   # model already selected as best
        workers=workers,
        timeout=timeout,
    )
