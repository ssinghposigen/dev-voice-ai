from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform
from typing import List, Dict
import boto3
import json

# Component for listing new transcripts
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["boto3", "google-cloud-storage"]
)
def list_new_transcripts(
    aws_access_key_id: str,
    aws_secret_key: str,
    source_bucket: str,
    time_window_hours: int = 2
) -> List[str]:
    import boto3
    from datetime import datetime, timedelta
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key
    )
    
    # List files modified in the last n hours
    time_threshold = datetime.utcnow() - timedelta(hours=time_window_hours)
    
    response = s3_client.list_objects_v2(Bucket=source_bucket)
    new_files = [
        obj['Key'] for obj in response.get('Contents', [])
        if obj['LastModified'].replace(tzinfo=None) > time_threshold
        and obj['Key'].endswith('.json')
    ]
    
    return new_files

# Component for processing single transcript
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "boto3",
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "pandas",
        "snowflake-connector-python"
    ]
)
def process_transcript(
    transcript_key: str,
    aws_access_key_id: str,
    aws_secret_key: str,
    source_bucket: str,
    gcp_project: str,
    gcp_location: str,
    destination_bucket: str,
    snowflake_credentials: dict
) -> str:
    import pandas as pd
    from google.cloud import storage, aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
    import snowflake.connector
    
    # Download and process transcript
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key
    )
    
    response = s3_client.get_object(
        Bucket=source_bucket,
        Key=transcript_key
    )
    transcript_data = json.loads(response['Body'].read().decode('utf-8'))
    
    # Process with Gemini
    aiplatform.init(project=gcp_project, location=gcp_location)
    model = aiplatform.GenerativeModel('gemini-pro')
    
    # Combine transcript text
    full_text = ' '.join([t['Content'] for t in transcript_data['Transcript']])
    
    response = model.generate_content(
        f"""Analyze this customer service conversation and provide:
        1. Topic
        2. Category
        3. Sentiment
        4. Key issues
        
        Conversation: {full_text}
        
        Respond in JSON format with these keys."""
    )
    
    analysis = json.loads(response.text)
    
    # Create DataFrame with analysis
    df = pd.DataFrame(transcript_data['Transcript'])
    for key, value in analysis.items():
        df[key] = value
    
    # Save to Snowflake
    with snowflake.connector.connect(**snowflake_credentials) as conn:
        success, nchunks, nrows, _ = write_pandas(
            conn,
            df,
            'processed_transcripts',
            auto_create_table=True
        )
    
    return f"Processed {transcript_key}: {nrows} rows written"

# Define the pipeline
@dsl.pipeline(
    name="transcript-processing-pipeline",
    description="Process AWS Connect transcripts with parallel execution"
)
def transcript_pipeline(
    aws_access_key_id: str,
    aws_secret_key: str,
    source_bucket: str,
    gcp_project: str,
    gcp_location: str,
    destination_bucket: str,
    snowflake_credentials: dict,
    max_parallel_executions: int = 10
):
    # List new transcripts
    list_task = list_new_transcripts(
        aws_access_key_id=aws_access_key_id,
        aws_secret_key=aws_secret_key,
        source_bucket=source_bucket
    )
    
    # Process transcripts in parallel with resource constraints
    with dsl.ParallelFor(
        items=list_task.output,
        parallelism=max_parallel_executions
    ) as transcript_key:
        process_transcript(
            transcript_key=transcript_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_key=aws_secret_key,
            source_bucket=source_bucket,
            gcp_project=gcp_project,
            gcp_location=gcp_location,
            destination_bucket=destination_bucket,
            snowflake_credentials=snowflake_credentials
        )

# Compile and deploy the pipeline
def deploy_pipeline():
    pipeline_spec = "transcript_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=transcript_pipeline,
        package_path=pipeline_spec
    )
    
    # Create pipeline job
    aiplatform.init(
        project="your-project",
        location="your-location"
    )
    
    job = aiplatform.PipelineJob(
        display_name="transcript-processing",
        template_path=pipeline_spec,
        pipeline_root="gs://your-bucket/pipeline_root",
        parameter_values={
            "aws_access_key_id": "your-key",
            "aws_secret_key": "your-secret",
            "source_bucket": "your-source-bucket",
            "gcp_project": "your-project",
            "gcp_location": "your-location",
            "destination_bucket": "your-destination-bucket",
            "snowflake_credentials": {
                "user": "your-user",
                "password": "your-password",
                "account": "your-account",
                "warehouse": "your-warehouse",
                "database": "your-database",
                "schema": "your-schema"
            },
            "max_parallel_executions": 10
        }
    )
    
    job.run()

# Schedule pipeline execution
def create_schedule():
    aiplatform.PipelineJob.schedule(
        display_name="transcript-processing-schedule",
        pipeline_file_path="transcript_pipeline.json",
        schedule="0 */2 * * *",  # Every 2 hours
        time_zone="UTC",
        parameter_values={...}  # Same as above
    )
