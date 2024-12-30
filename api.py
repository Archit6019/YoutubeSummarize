import gradio as gr 
from service import AI
import asyncio

async def process_video_url(api_key: str, video_url: str):
    try:
        ai_model = AI(api_key=api_key)
        summary = await ai_model.main(video_url)
        return summary
    except Exception as e:
        return f"Error processing video: {e}"

def wrap_async(func):
    def sync_func(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return sync_func

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Video Summarizer")
    gr.Markdown(
        "Paste your OpenAI API key and a YouTube URL below to get a concise summary of the video's transcript!"
    )

    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API key...",
                lines=1,
                type="password",
            )
            video_url = gr.Textbox(
                label="YouTube Video URL",
                placeholder="Enter a YouTube video URL...",
                lines=1,
            )
            submit_button = gr.Button("Summarize")
        with gr.Column():
            output = gr.Textbox(
                label="Summary",
                placeholder="Summary will appear here...",
                lines=10,
            )

    submit_button.click(
        fn=wrap_async(process_video_url),
        inputs=[api_key, video_url],
        outputs=output,
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)