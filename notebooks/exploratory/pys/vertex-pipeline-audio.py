from kfp import dsl
from kfp.v2 import compiler
from google.cloud import storage, speech_v1
from google.cloud import aiplatform
import pandas as pd
import boto3
import os
from typing import NamedTuple

# Component 1: Load audio from AWS S3
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "boto3",
        "google-cloud-storage"
    ]
)
def download_from_s3(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket: str,
    s3_prefix: str,
    gcs_temp_bucket: str,
    gcs_temp_prefix: str
) -> NamedTuple('Outputs', [
    ('audio_uri', str)
]):
    import boto3
    from google.cloud import storage
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    # Initialize GCS client
    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(gcs_temp_bucket)
    
    # Download from S3 and upload to GCS
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix)
    audio_data = response['Body'].read()
    
    gcs_blob = gcs_bucket.blob(f"{gcs_temp_prefix}/audio_file.wav")
    gcs_blob.upload_from_string(audio_data)
    
    audio_uri = f"gs://{gcs_temp_bucket}/{gcs_temp_prefix}/audio_file.wav"
    
    return (audio_uri,)

# Component 2: Transcribe Audio
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-speech",
        "google-cloud-storage"
    ]
)
def transcribe_audio(
    audio_uri: str,
    gcs_output_bucket: str,
    gcs_output_prefix: str
) -> NamedTuple('Outputs', [
    ('transcript_uri', str)
]):
    from google.cloud import speech_v1
    
    client = speech_v1.SpeechClient()
    
    audio = speech_v1.RecognitionAudio(uri=audio_uri)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()
    
    # Combine all transcriptions
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + "\n"
    
    # Save transcript to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_output_bucket)
    blob = bucket.blob(f"{gcs_output_prefix}/transcript.txt")
    blob.upload_from_string(transcript)
    
    transcript_uri = f"gs://{gcs_output_bucket}/{gcs_output_prefix}/transcript.txt"
    return (transcript_uri,)

# Component 3: KPI Extraction
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas",
        "google-cloud-storage",
        "transformers",
        "torch"
    ]
)
def extract_kpis(
    transcript_uri: str,
    gcs_output_bucket: str,
    gcs_output_prefix: str
) -> NamedTuple('Outputs', [
    ('kpi_results_uri', str)
]):
    from transformers import pipeline
    import pandas as pd
    from google.cloud import storage
    
    # Download transcript from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_output_bucket)
    blob = bucket.blob(transcript_uri.split(f"gs://{gcs_output_bucket}/")[1])
    transcript = blob.download_as_text()
    
    # Initialize NLP pipelines
    summarizer = pipeline("summarization")
    classifier = pipeline("zero-shot-classification")
    
    # Extract KPIs
    kpis = {
        'transcript': transcript,
        'summary': summarizer(transcript, max_length=130, min_length=30)[0]['summary_text'],
        'topic': classifier(
            transcript,
            candidate_labels=["sales", "support", "technical", "billing", "general inquiry"]
        )['labels'][0],
        'sub_topic': classifier(
            transcript,
            candidate_labels=["product information", "pricing", "complaints", "feedback", "other"]
        )['labels'][0]
    }
    
    # Create DataFrame
    df = pd.DataFrame([kpis])
    
    # Save to GCS
    output_path = f"{gcs_output_prefix}/kpi_results.csv"
    blob = bucket.blob(output_path)
    blob.upload_from_string(df.to_csv(index=False))
    
    kpi_results_uri = f"gs://{gcs_output_bucket}/{output_path}"
    return (kpi_results_uri,)

# Define the pipeline
@dsl.pipeline(
    name="audio-processing-pipeline",
    description="Pipeline to process audio files and extract KPIs"
)
def audio_processing_pipeline(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket: str,
    s3_prefix: str,
    gcs_temp_bucket: str,
    gcs_output_bucket: str
):
    # Step 1: Download from S3
    download_task = download_from_s3(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        gcs_temp_bucket=gcs_temp_bucket,
        gcs_temp_prefix="temp_audio"
    )
    
    # Step 2: Transcribe
    transcribe_task = transcribe_audio(
        audio_uri=download_task.outputs['audio_uri'],
        gcs_output_bucket=gcs_output_bucket,
        gcs_output_prefix="transcripts"
    )
    
    # Step 3: Extract KPIs
    kpi_task = extract_kpis(
        transcript_uri=transcribe_task.outputs['transcript_uri'],
        gcs_output_bucket=gcs_output_bucket,
        gcs_output_prefix="kpi_results"
    )

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=audio_processing_pipeline,
    package_path='audio_processing_pipeline.json'
)
