from typing import List, Dict
from openai import AsyncOpenAI
import logging
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed, formatters
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AI:
    def __init__(self, api_key: str):
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.formatter = formatters.TextFormatter()
    
    async def get_transcript(self, video_url: str) -> str:
        try:
            video_id = extract.video_id(video_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text_formatted = self.formatter.format_transcript(transcript=transcript)
            return text_formatted
        except YouTubeRequestFailed as e:
            logger.error(f"Youtube Request Failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            raise

    async def get_summary(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> str:
        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in getting response: {e}")
            raise

    async def main(self, video_url: str) -> str:
        try:
            transcript = await self.get_transcript(video_url)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant, tasked with creating short concise summaries of youtube transcripts, from the transcripts try to grasp the main idea and generate summary around it"
                },
                {
                    "role": "user",
                    "content": f"Transcript: {transcript}"
                }
            ]
            summary = await self.get_summary(messages)
            return summary

        except Exception as e:
            logger.error(f"Exception in main: {e}")
            traceback.print_exc()
            raise