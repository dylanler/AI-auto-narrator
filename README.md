# README for Auto Narrator Project

## Overview

Auto Narrator is a Python application designed to automate the process of generating voiceover narrations for videos. It utilizes a combination of video processing, text-to-speech technology, and machine learning models to produce narrations based on the content of the video and user-defined prompts.

## Demo Video

For a demonstration of the Auto Narrator application in action, watch our video here: [Auto Narrator Demo](https://www.youtube.com/watch?v=eP58jyDNor8)

https://github.com/dylanler/AI-auto-narrator/assets/9219358/0ad03bf2-5905-43bd-97c0-91b849a5511f

## Prerequisites

- Python 3.6 or later
- OpenAI API key
- Eleven Labs API key
- Basic knowledge of Python and command line usage

## Installation

1. **Clone the Repository:**
   Clone this repository to your local machine using `git clone`.

2. **Install Dependencies:**
   Run `pip install -r requirements.txt` to install the required Python libraries.

3. **Set Up Environment Variables:**
   Set the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `ELEVEN_LABS_API`: Your Eleven Labs API key.
   - `PASSWORD_AUTH`: A password for Gradio web interface authentication.

## Usage

### Running the Application

1. Start the application by running `python script_name.py` (replace `script_name.py` with the actual name of the script).
2. Navigate to the displayed URL to access the Gradio web interface.
3. Follow the on-screen instructions to upload a video and provide narration details.

### Generating Narration

1. **Upload a Video:**
   Select a video file to upload for narration.

2. **Enter Your Prompt:**
   Enter a custom prompt or select a predefined prompt type (e.g., documentary, how-to).

3. **Choose Voice Type:**
   Select a voice type for the narration (e.g., feminine-american, masculine-british).

4. **Generate Narration:**
   Click the "Generate" button to process the video and generate the narration.

5. **Review Output:**
   The processed video with narration and the narration script will be displayed.

### Regenerating Narration

1. Modify the generated narration script as needed.
2. Click the "Re-generate" button to apply the changes and regenerate the narration.

## Functions Description

- `video_to_frames(video_file_path)`: Converts a video file to a series of base64 encoded frames.
- `text_to_speech(text, video_filename, voice_type)`: Converts text to speech using the Eleven Labs API.
- `frames_to_story(base64Frames, prompt, video_duration)`: Generates a story or narration based on video frames and a prompt using OpenAI's model.
- `prompt_type(prompt_user, prompt_input, video_duration)`: Constructs a prompt for the AI model based on user input and video duration.
- `merge_audio_video(video_filename, audio_filename, output_filename)`: Merges the generated audio narration with the video.
- `process_video(uploaded_file, prompt_user, prompt_input, voice_type)`: Main function to process the video and generate narration.
- `regenerate(uploaded_file, edited_script, voice_type)`: Regenerates the narration based on an edited script.

## Notes

- Ensure you have valid API keys for OpenAI and Eleven Labs.
- This application is for educational and development purposes. Check the API providers' terms of service for commercial use.

## Support

For any queries or issues, please open an issue on the GitHub repository page.

---

*This README provides a comprehensive guide to setting up and using the Auto Narrator application. Adjust the instructions based on the specific requirements and configurations of your project.*

# Docker Deployment

## Running with Docker Compose

1. **Set up environment variables:**
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ELEVEN_LABS_API=your_eleven_labs_key_here
   PASSWORD_AUTH=your_password_here
   ```

2. **Start the application:**
   ```bash
   docker-compose up
   ```

3. **Access the application:**
   Once running, access the application at `http://localhost:7860`

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

## Environment Variables

The following environment variables are required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ELEVEN_LABS_API`: Your Eleven Labs API key
- `PASSWORD_AUTH`: Password for the Gradio interface (optional)

You can set these in the `.env` file or pass them directly when running docker-compose:
```bash
OPENAI_API_KEY=key1 ELEVEN_LABS_API=key2 PASSWORD_AUTH=pass docker-compose up
```