import os
import gradio as gr
from voice_of_the_patient import record_audio, transcribe_with_groq
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_doctor import text_to_speech_with_elevenlabs

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response.If the user gives his/her prescription of another doctor then analyze it. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, avoid any numbers or special characters in 
            your response, Use precise medical terms, Avoid metaphors or unrelated references, Validate responses against medical guidelines, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 10 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    try:
        # Transcribe audio
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )

        # Analyze image
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath) if image_filepath else None,
            model="llama-3.2-11b-vision-preview"
        ) if image_filepath else "No image provided"

        # Generate TTS
        text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath="final.mp3"
        )

        return speech_to_text_output, doctor_response, "final.mp3"
    except Exception as e:
        return f"Error: {str(e)}", "", ""

with gr.Blocks(title="AI Doctor with Vision and Voice") as app:
    gr.Markdown("# 🩺 AI Doctor (Vision & Voice)")
    
    with gr.Row():
        record_btn = gr.Button("🎤 Start Recording", variant="primary")
        audio_status = gr.Textbox(label="Recording Status", interactive=False)
        audio_path = gr.Textbox(visible=False)  
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Medical Image")
        submit_btn = gr.Button("🚀 Analyze", variant="primary")
    
    with gr.Column():
        gr.Markdown("## Results")
        stt_output = gr.Textbox(label="Patient's Speech Transcription")
        diagnosis_output = gr.Textbox(label="Doctor's Diagnosis")
        tts_output = gr.Audio(label="Voice Response", type="filepath")

    # Fixed recording handler
    def handle_recording():
        try:
            record_audio("patient_recording.mp3")
            return "Recording saved!", "patient_recording.mp3"
        except Exception as e:
            return f"Recording failed: {str(e)}", None

    record_btn.click(
        fn=handle_recording,
        outputs=[audio_status, audio_path]
    )

    # Submit handler
    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_path, image_input],
        outputs=[stt_output, diagnosis_output, tts_output]
    )

if __name__ == "__main__":
    app.launch(debug=True)