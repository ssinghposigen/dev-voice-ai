from google.cloud import speech
from google.cloud.speech_v1 import SpeechClient
from google.cloud.speech_v1.types import RecognitionConfig, SpeakerDiarizationConfig
from datetime import datetime, timedelta

def transcribe_with_diarization(audio_uri):
    """
    Transcribe audio with speaker diarization and timing information
    
    Args:
        audio_uri (str): URI of the audio file to transcribe
        
    Returns:
        list: List of dictionaries containing speaker segments with timing information
    """
    client = speech.SpeechClient()

    speaker_diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        sample_rate_hertz=8000,
        enable_word_time_offsets=True,  # Enable word-level timing
        diarization_config=speaker_diarization_config,
    )

    audio = speech.RecognitionAudio(
        uri=audio_uri,
    )

    # Perform the transcription
    response = client.long_running_recognize(
        config=recognition_config,
        audio=audio,
    ).result()

    # Process results to get speaker segments with timing
    speaker_segments = []
    current_speaker = None
    segment_start = None
    current_text = []

    for result in response.results:
        # Skip results without speaker tags
        if not result.alternatives[0].words:
            continue

        words = result.alternatives[0].words

        for word in words:
            word_start_time = word.start_time.total_seconds()
            word_end_time = word.end_time.total_seconds()
            speaker_tag = word.speaker_tag

            # Start new segment if speaker changes or this is the first word
            if speaker_tag != current_speaker:
                # Save previous segment if it exists
                if current_speaker is not None:
                    speaker_segments.append({
                        'speaker': f'Speaker {current_speaker}',
                        'start_time': format_timestamp(segment_start),
                        'end_time': format_timestamp(word_start_time),
                        'text': ' '.join(current_text)
                    })

                # Start new segment
                current_speaker = speaker_tag
                segment_start = word_start_time
                current_text = [word.word]
            else:
                current_text.append(word.word)

        # Don't forget to add the last segment
        if current_text:
            speaker_segments.append({
                'speaker': f'Speaker {current_speaker}',
                'start_time': format_timestamp(segment_start),
                'end_time': format_timestamp(word_end_time),
                'text': ' '.join(current_text)
            })

    return speaker_segments

def format_timestamp(seconds):
    """
    Format seconds into HH:MM:SS.mmm
    """
    time = str(timedelta(seconds=seconds))
    # Ensure consistent formatting
    if '.' not in time:
        time += '.000'
    else:
        time = time[:-3]  # Truncate to milliseconds
    # Pad hours if necessary
    if len(time.split(':')[0]) == 1:
        time = '0' + time
    return time

# Example usage
if __name__ == "__main__":
    audio_uri = "gs://your-bucket/your-audio-file.wav"
    
    try:
        segments = transcribe_with_diarization(audio_uri)
        
        # Print formatted results
        for segment in segments:
            print(f"\n{segment['speaker']}")
            print(f"Start Time: {segment['start_time']}")
            print(f"End Time: {segment['end_time']}")
            print(f"Text: {segment['text']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
