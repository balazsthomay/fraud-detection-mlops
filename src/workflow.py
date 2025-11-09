from prefect import flow, task
from src.train_pipeline import main as train_pipeline
from src.compare_and_deploy import main as compare_and_deploy


@task(log_prints=True)
def train_pipeline_task():
    train_pipeline()
    
@task(log_prints=True)
def compare_and_deploy_task():
    compare_and_deploy()

@flow(log_prints=True)
def main_workflow():
    train_pipeline_task()
    compare_and_deploy_task()
    print("Trained, compared, and deployed!")


if __name__ == "__main__":
    main_workflow.serve(
        name="fraud-detection-pipeline",
        cron="*/2 * * * *"  # Every 2 minutes for testing
    )
    
# if __name__ == "__main__":
#     main_workflow.serve(
#         name="fraud-detection-pipeline",
#         cron="0 2 * * 0"  # Every Sunday at 2am
#     )    
    
    
# before you run, start prefect server in another terminal:
# prefect server start

# then:
# python -m src.workflow
    
# * * * * *
# │ │ │ │ │
# │ │ │ │ └─ Day of week (0-7, where 0 and 7 = Sunday)
# │ │ │ └─── Month (1-12)
# │ │ └───── Day of month (1-31)
# │ └─────── Hour (0-23)
# └───────── Minute (0-59)