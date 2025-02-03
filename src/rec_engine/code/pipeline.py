import sys
from pathlib import Path
from typing import Any, Dict, Union

from azureml.core import Experiment, ScriptRunConfig, Workspace
from azureml.core.environment import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

from config import ParametersConfig, get_toml
from logger import LoggerConfig

# Configure logging
logger = LoggerConfig.configure_logger("Pipeline")


class PR_Pipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.args = ParametersConfig.from_toml(path_or_dict=self.config)
        self.ws = Workspace.from_config()

    @classmethod
    def from_toml(
        cls, path_or_dict: Union[Path, str, Dict[str, Any]]
    ) -> "Pipeline":
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            config = get_toml(path_or_dict)
        return cls(config=config)

    def create_experiment(self, experiment_name: str) -> Experiment:
        """
        Create an Azure ML experiment
        """
        experiment = Experiment(workspace=self.ws, name=experiment_name)
        logger.info(f"Experiment {experiment_name} created")
        return experiment

    def get_or_create_environment(
        self, environment_name: str, environment_file: str
    ) -> Environment:
        """
        Get the environment if it exists, otherwise create it from the environment.yml file
        """
        try:
            env = Environment.get(workspace=self.ws, name=environment_name)
            logger.info(f"Environment {environment_name} already exists")
            # Update the environment if the environment file has changed
            # Load the environment from the environment.yml file
            new_env = Environment.from_conda_specification(
                name=environment_name, file_path=environment_file
            )

            # Compare the dependencies
            if (
                env.python.conda_dependencies.serialize_to_string()
                != new_env.python.conda_dependencies.serialize_to_string()
            ):
                logger.info(
                    f"Environment {environment_name} differs from the specification in {environment_file}, updating it"
                )
                env = new_env
                env.register(workspace=self.ws)
                logger.info(
                    f"Environment {environment_name} updated and registered"
                )

        except Exception as e:
            logger.info(
                f"Environment {environment_name} does not exist, creating it from {environment_file}"
            )
            env = Environment.from_conda_specification(
                name=environment_name, file_path=environment_file
            )
            env.register(workspace=self.ws)
            logger.info(
                f"Environment {environment_name} created and registered"
            )
        return env

    def submit_job(
        self,
        experiment: Experiment,
        script_path: str,
        environment_name: str,
        environment_file: str,
    ) -> None:
        """
        Submit a job to the Azure ML experiment
        """
        # Get or create a new environment
        env = self.get_or_create_environment(environment_name, environment_file)

        # Create a new run configuration
        run_config = RunConfiguration()
        run_config.environment = env

        # Create a script run configuration
        src = ScriptRunConfig(
            source_directory="src/rec_engine/code",
            script=script_path,
            run_config=run_config,
        )

        # Submit the job
        run = experiment.submit(src)
        logger.info(f"Job submitted to experiment {experiment.name}")
        run.wait_for_completion(show_output=True)
        logger.info(f"Job completed with status {run.get_status()}")

    def create_pipeline(
        self,
        experiment: Experiment,
        environment_name: str,
        environment_file: str,
    ) -> Pipeline:
        """
        Create an Azure ML pipeline
        """
        # Get or create the environment
        env = self.get_or_create_environment(environment_name, environment_file)

        # Create a new run configuration
        run_config = RunConfiguration()
        run_config.environment = env

        # Set src dir to PYTHONPATH environment variable
        run_config.environment.environment_variables = {
            "PYTHONPATH": str(src_path)
        }
        # Define pipeline data
        raw_data = PipelineData(
            "outputs", datastore=self.ws.get_default_datastore()
        )

        # Define the steps
        data_loader_step = PythonScriptStep(
            name="Data Loader",
            source_directory=str(src_path),
            script_name="rec_engine/code/data_loader.py",
            arguments=["--output", raw_data],
            outputs=[raw_data],
            compute_target="CI-usmushtaq-DS13-V2-prd",
            runconfig=run_config,
            allow_reuse=False,
        )

        data_processor_step = PythonScriptStep(
            name="Data Processor",
            source_directory=str(src_path),
            script_name="rec_engine/code/data_processor.py",
            arguments=["--input", raw_data],
            inputs=[raw_data],
            compute_target="CI-usmushtaq-DS13-V2-prd",
            runconfig=run_config,
            allow_reuse=False,
        )
        # Create the pipeline
        pipeline = Pipeline(
            workspace=self.ws, steps=[data_loader_step, data_processor_step]
        )
        logger.info("Pipeline created")
        return pipeline

    def submit_pipeline(
        self, experiment: Experiment, pipeline: Pipeline
    ) -> None:
        """
        Submit the pipeline to the Azure ML experiment
        """
        pipeline_run = experiment.submit(pipeline)
        logger.info(f"Pipeline submitted to experiment {experiment.name}")
        pipeline_run.wait_for_completion(show_output=True)
        logger.info(
            f"Pipeline completed with status {pipeline_run.get_status()}"
        )


if __name__ == "__main__":
    config = get_toml()
    pipeline = PR_Pipeline.from_toml(config)

    # Create an experiment
    experiment_name = "data_processing_experiment"
    experiment = pipeline.create_experiment(experiment_name)

    # Submit data loader job
    data_loader_script = "data_loader.py"
    environment_name = "usman_test_environment"
    environment_file = "environment/environment.yml"
    data_pipeline = pipeline.create_pipeline(
        experiment, environment_name, environment_file
    )
    pipeline.submit_pipeline(experiment, data_pipeline)
