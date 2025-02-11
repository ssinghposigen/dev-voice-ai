import boto3
from botocore.exceptions import ClientError
import google.generativeai as genai
import os
from io import BytesIO

def get_s3_audio_uri(bucket_name: str, object_key: str, aws_access_key: str, aws_secret_key: str) -> BytesIO:
    """
    Fetch audio file from S3 and return it as a BytesIO object
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Get the audio file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        audio_data = response['Body'].read()
        
        # Convert to BytesIO object
        return BytesIO(audio_data)
        
    except ClientError as e:
        print(f"Error accessing S3: {e}")
        raise

def transcribe_audio(audio_data: BytesIO, api_key: str) -> str:
    """
    Transcribe audio using Gemini 1.5 Pro
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 1.5 Pro model
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            generation_config={
                'temperature': 0.1,  # Lower temperature for more accurate transcription
                'top_p': 0.8,
                'top_k': 40,
            }
        )
        
        # Prepare audio content
        audio_content = audio_data.getvalue()
        
        # Generate transcription
        response = model.generate_content(
            audio_content,
            stream=False,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        raise

def main():
    # Configuration
    AWS_ACCESS_KEY = "your_aws_access_key"
    AWS_SECRET_KEY = "your_aws_secret_key"
    BUCKET_NAME = "your_bucket_name"
    OBJECT_KEY = "path/to/your/audio/file.mp3"
    GEMINI_API_KEY = "your_gemini_api_key"
    
    try:
        # Get audio data from S3
        audio_data = get_s3_audio_uri(
            bucket_name=BUCKET_NAME,
            object_key=OBJECT_KEY,
            aws_access_key=AWS_ACCESS_KEY,
            aws_secret_key=AWS_SECRET_KEY
        )
        
        # Transcribe audio
        transcription = transcribe_audio(
            audio_data=audio_data,
            api_key=GEMINI_API_KEY
        )
        
        print("Transcription:", transcription)
        
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main()
