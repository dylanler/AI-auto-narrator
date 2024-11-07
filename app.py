import os
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
ELEVEN_LABS_API = os.environ['ELEVEN_LABS_API']
PASSWORD_AUTH = os.environ['PASSWORD_AUTH']

from elevenlabs.api import Voice  # for newer versions
from elevenlabs import generate, save, voices, clone
from elevenlabs import set_api_key
set_api_key(ELEVEN_LABS_API)

def process_video_custom_voice(uploaded_file, prompt_user, prompt_input, custom_audio, voice_prompt, model):
    
    if type(uploaded_file) == str:
        video_filename = uploaded_file
    else:
        video_filename = uploaded_file.name
    print("video", video_filename)
        
    base64Frames, video_filename, video_duration = video_to_frames(video_filename)

    final_prompt = prompt_type(prompt_user, prompt_input, video_duration)
    print(final_prompt)
    text = frames_to_story(base64Frames, final_prompt, video_duration, model)
    
    if type(custom_audio) == str:
        custom_audio_filename = custom_audio
    else:
        custom_audio_filename = custom_audio.name
    print("custom audio", custom_audio_filename)

    voice = clone(
        name="Custom Voice",
        description=f"{voice_prompt}", # Optional
        files=[custom_audio_filename],
    )

    audio = generate(text=text, voice=voice)
    save(audio, custom_audio_filename)
    
    audio_filename = custom_audio_filename

    # Merge audio and video
    output_video_filename = os.path.splitext(video_filename)[0] + '_output.mp4'
    final_video_filename = merge_audio_video(video_filename, audio_filename, output_video_filename)
    print("final", final_video_filename)

    if type(uploaded_file) != str:
        os.unlink(video_filename)
        os.unlink(audio_filename)
    
    return final_video_filename, text


import openai
import requests
import os
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import cv2  # We're using OpenCV to read video
import base64
import time
import io
import tempfile
import numpy as np
import gradio as gr



# Set your OpenAI API key here
openai.api_key = OPENAI_API_KEY

def video_to_frames(video_file_path):
    
    if type(video_file_path) == str:
        video_filename = video_file_path
    else:
        video_filename = video_file_path.name


    video_duration = VideoFileClip(video_filename).duration

    video = cv2.VideoCapture(video_filename)
    base64Frames = []
    
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
        if frame_count % 30 == 0:
            print("30 frames added.")

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames, video_filename, video_duration


def text_to_speech(text, video_filename, voice_type="feminine-american", API_KEY = ELEVEN_LABS_API):
    
    CHUNK_SIZE = 2048
    voice_id = '21m00Tcm4TlvDq8ikWAM'
    BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    
    if voice_type == "masculine-american":
        
        MODEL_ID = "eleven_monolingual_v1"
        voice_id = 'VR6AewLTigWG4xSOukaG'
        BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        chunk = text
        data = {
            "text": chunk,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
    elif voice_type == "feminine-british":
        
        MODEL_ID = "eleven_monolingual_v1"
        voice_id = 'ThT5KcBeYPX3keUQqHPh'
        BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        chunk = text
        data = {
            "text": chunk,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
    elif voice_type == "masculine-british":
        
        MODEL_ID = "eleven_monolingual_v1"
        voice_id = 'Yko7PKHZNXotIFUBG7I9'
        BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        chunk = text
        data = {
            "text": chunk,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
    else:
        
        MODEL_ID = "eleven_monolingual_v1"
        voice_id = 'jsCqWAovK2LkecY7zXl4'
        BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        chunk = text
        data = {
            "text": chunk,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.5
            }
        }

    # Send the POST request to the API
    response = requests.post(BASE_URL, json=data, headers=headers)

    # Check if the response is OK
    if response.status_code == 200:
        # Write the chunk to an mp3 file in the directory
        # Save audio to a specified file
        audio_filename = 'testing_file.mp3'
        with open(audio_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)

        print(f'Saved {audio_filename}')
    else:
        print(f'Error: Received response code {response.status_code}')

    return audio_filename

def frames_to_story(base64Frames, prompt, video_duration, model):
    
    fps = int(len(base64Frames) / video_duration)
    
    frame_cut_thres = fps
    print("Cutting at", frame_cut_thres)
    
    list_of_dictionaries = list(map(lambda x: {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{x}",
            "detail": "low"
        }
    }, base64Frames[0::frame_cut_thres])) 
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *list_of_dictionaries,
            ],
        },
    ]
    params = {
        "model": model,
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
        
    }

    result = openai.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


def prompt_type(prompt_user, prompt_input, video_duration):

    prompt_documentary = '''
    You are a world class documentary narration script writer.
    Based on the frames in the video, write a captivating voiceover for it.
    Write it with close observation of each frame.
    Observe the suddent change in movement of each frame and narrate about it.
    '''

    prompt_how_to = '''
    You are an expert narrator that specializes in writing narration scripts for "how-to" videos.
    Your goal is to write a script so that the audience can follow instructions from the video.
    Pay attention to where the mouse and tap cursor is and navigate based on the sequence of each frame.
    Remember to narrate something useful.  Narrate something that the audience can understand to take an action.
    '''

    prompt_sports_commentator = '''
    You are a professional sports commentator that can comment for all kinds of sports including e-sports.
    Your goal is to write a script that is exciting and make the audience's heart beat fast.
    Pay attention to what the characters of the players are doing in each frame and narrate their actions.
    Remember to narrate something exciting and nail-biting.  Keep the audience on their toes and wanting to know more.
    Add a lot of exclamation mark and emotions into the voiceover script.
    '''
    
    if prompt_input == "how-to":
        prompt_input = prompt_how_to
        mul_factor = 1.6
    elif prompt_input == "documentary":
        prompt_input = prompt_documentary
        mul_factor = 2
    elif prompt_input == "sports-commentator":
        prompt_input = prompt_sports_commentator
        mul_factor = 1.5
    elif prompt_input == "custom-prompt":
        prompt_input = prompt_user
        mul_factor = 2
    else:
        prompt_input = ""
        mul_factor = 2

    est_word_count = int(video_duration * mul_factor)
    
    word_lim_prompt = f'''This video is EXACTLY {video_duration} seconds long, 
    make sure the voiceover narration script to be EXACTLY {est_word_count} words. 
    Do not go over {est_word_count} for the output script.
    '''

    initial_prompt = '''
    These are a sequence of frames for a short video.
    You are an expert voiceover script writer.  The voiceover is to help the audience and viewer.
    Write a voiceover for the video by carefully analyzing each frame.
    Make sure there is coherence between each frame.
    '''
    final_prompt = word_lim_prompt + initial_prompt + prompt_user + prompt_input + "\n" + word_lim_prompt
    
    return(final_prompt)


def merge_audio_video(video_filename, audio_filename, output_filename, original_audio_volume=0.3):
    print("Merging audio and video...")
    print("Video filename:", video_filename)
    print("Audio filename:", audio_filename)

    # Load the video file
    video_clip = VideoFileClip(video_filename)

    try:# Reduce the volume of the original audio
        original_audio = video_clip.audio.volumex(original_audio_volume)
        
        # Load the new audio file
        new_audio_clip = AudioFileClip(audio_filename)

        # Mix the adjusted original audio with the new audio
        mixed_audio = CompositeAudioClip([original_audio, new_audio_clip])

        # Set the mixed audio as the audio of the video clip
        final_clip = video_clip.set_audio(mixed_audio)

        # Write the result to a file
        final_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')

        # Close the clips
        video_clip.close()
        new_audio_clip.close()
        
    except:
        print("No volume")
        
        # Set the audio of the video clip
        final_clip = video_clip.set_audio(audio_filename)

        # Write the result to a file
        final_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')

        # Close the clips
        video_clip.close()
        new_audio_clip.close()
        

    # Return the path to the new video file
    return output_filename



# Rest of your imports and functions remain the same

def process_video(uploaded_file, prompt_user, prompt_input, voice_type="feminine-american", model):
    if type(uploaded_file) == str:
        video_filename = uploaded_file
    else:
        video_filename = uploaded_file.name
    print("video", video_filename)
        
    base64Frames, video_filename, video_duration = video_to_frames(video_filename)

    final_prompt = prompt_type(prompt_user, prompt_input, video_duration)
    print(final_prompt)
    text = frames_to_story(base64Frames, final_prompt, video_duration, model)

    audio_filename = text_to_speech(text, video_filename, voice_type)
    print("audio", audio_filename)

    # Merge audio and video
    output_video_filename = os.path.splitext(video_filename)[0] + '_output.mp4'
    final_video_filename = merge_audio_video(video_filename, audio_filename, output_video_filename)
    print("final", final_video_filename)

    if type(uploaded_file) != str:
        os.unlink(video_filename)
        os.unlink(audio_filename)
    
    return final_video_filename, text

# Rest of your imports and functions remain the same

def regenerate(uploaded_file, edited_script, voice_type="feminine-american"):
    
    if type(uploaded_file) == str:
        video_filename = uploaded_file
    else:
        video_filename = uploaded_file.name
    print("video", video_filename)
    
    # Generate audio from text
    audio_filename = text_to_speech(edited_script, video_filename, voice_type)
    print("audio", audio_filename)

    # Merge audio and video
    output_video_filename = os.path.splitext(video_filename)[0] + '_output.mp4'
    final_video_filename = merge_audio_video(video_filename, audio_filename, output_video_filename)
    print("final", final_video_filename)

    if type(uploaded_file) != str:
        os.unlink(video_filename)
        os.unlink(audio_filename)
    
    return final_video_filename, edited_script

with gr.Blocks() as demo:
    
    gr.Markdown(
    """
    # Auto Narrator
    Upload a video and provide a prompt to generate a narration.
    """)
    with gr.Row():
        with gr.Column():

            video_input = gr.Video(label="Upload Video")
            prompt_user = gr.Textbox(label="Enter your prompt")
            prompt_input = gr.Dropdown(['how-to', 'documentary', 'sports-commentator', 'custom-prompt'], label="Choose Your Narration")
            openai_model = gr.Dropdown(
                choices=['gpt-4-vision-preview', 'gpt-4v'], 
                value='gpt-4-vision-preview',
                label="OpenAI image recognition model"
            )
            voice_type = gr.Dropdown(['masculine-american', 'masculine-british', 'feminine-american', 'feminine-british'], label="Choose Your Voice")
            
            generate_btn = gr.Button(value="Generate")
            voice_sample = gr.File(label="Use custom made voice.")
            voice_prompt = gr.Textbox(label="Enter voice prompt.")
            
            #render_btn = gr.Button(value="Render")
            #print_btn = gr.Button(value="Print")
        with gr.Column():
           
            output_file = gr.Video(label="Ouput video file.")
            output_voiceover = gr.Textbox(label="Generated Text")
            regenerate_btn = gr.Button(value="Re-generate")
            custom_voice_btn = gr.Button(value="Use Custom Voice")
            #print_text = gr.Text(label="Printing")

   
    generate_btn.click(process_video, inputs=[video_input, prompt_user, prompt_input, voice_type, openai_model], outputs=[output_file,output_voiceover])
    regenerate_btn.click(regenerate, inputs=[video_input, output_voiceover, voice_type], outputs=[output_file,output_voiceover])
    custom_voice_btn.click(process_video_custom_voice, inputs=[video_input, prompt_user, prompt_input, voice_sample, voice_prompt, openai_model], outputs=[output_file,output_voiceover])

    
if __name__ == "__main__":
    # Get password from environment variable, default to admin if not set
    password = os.getenv("PASSWORD_AUTH", "admin")
    demo.launch(
        server_name="0.0.0.0",  # Required for Docker
        server_port=7860,       # Standard Gradio port
        auth=("admin", password) if password else None
    )
