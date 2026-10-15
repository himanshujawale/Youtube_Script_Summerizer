import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import torch
import gradio as gr
from transformers import pipeline
from urllib.parse import urlparse, parse_qs

# text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)
#
# # model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots"
# #               "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# # text_summary = pipeline("summarization", model=model_path,
# #                 torch_dtype=torch.bfloat16)
#
# def summary (input):
#     output = text_summary(input)
#     return output[0]['summary_text']
text_summary_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    torch_dtype=torch.float32
)

def summarize_text(input_text):
    if not input_text or not input_text.strip():
        return "No text available to summarize."

    result = text_summary_model(
        input_text,
        max_length=150,
        min_length=30,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]
def extract_video_id(url):
    """Extract the YouTube video ID from different URL formats."""

    regex = (
        r"(?:https?://)?"
        r"(?:www\.)?"
        r"(?:youtube\.com/(?:watch\?.*?v=|embed/|v/|shorts/)|youtu\.be/)"
        r"([a-zA-Z0-9_-]{11})"
    )

    match = re.search(regex, url)

    if match:
        return match.group(1)

    return None


def get_youtube_transcript(url):
    try:
        # Extract the video ID from the URL
        video_id = extract_video_id(url)

        if not video_id:
            return "Invalid YouTube URL. Please enter a valid URL."

        # Fetch the transcript using the video ID
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        # Convert transcript into text
        transcript_text = " ".join(
            snippet.text for snippet in transcript
        )

        if not transcript_text.strip():
            return "No transcript available for this video."

        # Summarize the transcript
        summary = summarize_text(transcript_text)

        return summary

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"


# Example URL (Replace this with the actual URL when using the script)
# video_url = "https://youtu.be/5PibknhIsTc"
# print(get_youtube_transcript(video_url))

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=get_youtube_transcript,
                    inputs=[gr.Textbox(label="Input YouTube Url to summarize",lines=1)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="@Project 2: YouTube Script Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE YOUTUBE VIDEO SCRIPT.")
demo.launch()
