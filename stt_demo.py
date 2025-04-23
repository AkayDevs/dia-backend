import gradio as gr
from gradio import themes # Import themes explicitly
from transformers import pipeline
import torch
import time
import warnings
import os
import numpy as np
import io # For capturing logs
from contextlib import redirect_stdout, redirect_stderr # For capturing logs

# --- Configuration & Model Loading ---

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.functional')

# Define available models with user-friendly names and HF identifiers
AVAILABLE_MODELS = {
    # OpenAI Whisper Models
    "Whisper (Base - Multilingual)": "openai/whisper-base",
    "Whisper (Small - English)": "openai/whisper-small.en",
    "Whisper (Medium - English)": "openai/whisper-medium.en", # More accurate, needs more resources
    # Meta Wav2Vec2 Models
    "Wav2Vec2 (Base - English)": "facebook/wav2vec2-base-960h",
    # Add more models here if desired (e.g., multilingual Wav2Vec2/XLS-R)
    # "Wav2Vec2 (XLS-R 300M - Multilingual)": "facebook/wav2vec2-xls-r-300m",
}

# Cache for loaded pipelines to avoid reloading on every run
pipeline_cache = {}

# Determine device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# Initial log (will not be captured in UI, goes to terminal where script is launched)
print(f"--- Initializing ---")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Using device: {device}")


def load_pipeline(model_key):
    """Loads or retrieves a pipeline from cache. Raises exception on failure."""
    # This function's print statements WILL be captured if called within the 'with redirect...' block
    if model_key not in pipeline_cache:
        print(f"Attempting to load model: {model_key}...") # Log
        start_time = time.time()
        hf_model_id = AVAILABLE_MODELS[model_key]
        current_dtype = torch_dtype
        # Determine appropriate dtype based on model and device
        if "wav2vec2" in hf_model_id.lower() and device == "cpu":
             current_dtype = torch.float32
             print("Wav2Vec2 on CPU, forcing float32.") # Log
        elif device == "cpu":
            # General CPU usage might benefit from float32
            current_dtype = torch.float32
            print("Running on CPU, using float32.") # Log

        try:
            # Load the pipeline
            loaded_pipeline = pipeline(
                "automatic-speech-recognition",
                model=hf_model_id,
                torch_dtype=current_dtype,
                device=device
            )
            pipeline_cache[model_key] = loaded_pipeline # Cache the loaded pipeline
            end_time = time.time()
            print(f"Model '{model_key}' loaded successfully in {end_time - start_time:.2f} seconds.") # Log
            return loaded_pipeline
        except Exception as e:
            print(f"ERROR loading model {model_key}: {e}") # Log Error
            # Re-raise a more specific error to be caught and displayed
            raise RuntimeError(f"Failed to load model '{model_key}'. Error details: {e}") from e
    else:
        print(f"Using cached model: {model_key}") # Log
        return pipeline_cache[model_key]

# --- Transcription Function (Generator with Log Capture) ---

# --- Transcription Function (Generator with Incremental Log Capture) ---

def transcribe_audio(model_key, audio_input):
    """
    Transcribes audio, yields status updates, and captures/displays logs incrementally.
    Args:
        model_key (str): User-friendly name of the model selected.
        audio_input (tuple or str): Audio data from Gradio component.
    Yields:
        dict: Updates for Gradio components (status, output, logs).
    """
    status_updates = {}
    final_transcription = ""
    error_message = ""
    logs = ""
    # Use a list to accumulate logs incrementally for yielding
    log_accumulator = []
    log_stream = io.StringIO()

    try:
        # 1. Initial Status & Input Check
        status_updates[status_output] = gr.update(value="⏳ Initializing...", interactive=False)
        status_updates[output_textbox] = gr.update(value="", interactive=False)
        status_updates[log_output] = gr.update(value="") # Clear previous logs
        yield status_updates

        if audio_input is None:
             raise ValueError("No audio provided. Please upload a file or record audio.")

        # --- Start Capturing Logs ---
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            # 2. Load Model
            yield {status_output: gr.update(value=f"🔄 Loading model: {model_key}...")}
            asr_pipeline = load_pipeline(model_key) # This prints logs

            # --- >>> Incremental Log Update Point <<< ---
            logs_after_loading = log_stream.getvalue()
            log_accumulator.append(logs_after_loading)
            # Yield ONLY the log update after loading
            yield {log_output: gr.update(value="".join(log_accumulator))}
            # --- End Incremental Update ---

            # 3. Transcribe
            yield {status_output: gr.update(value="🎙️ Model loaded. Transcribing audio...")}

            print("\n--- Starting Transcription ---") # Add separator in log
            start_time = time.time()

            # Process audio input
            if isinstance(audio_input, tuple):
                sample_rate, audio_numpy = audio_input
                print(f"Processing numpy audio. SR: {sample_rate}, Shape: {audio_numpy.shape}")
                if audio_numpy is None or audio_numpy.size == 0: raise ValueError("Recorded/loaded audio is empty.")
                audio_numpy = audio_numpy.astype(np.float32)
                max_abs_val = np.max(np.abs(audio_numpy))
                if max_abs_val > 1.0:
                    print(f"Normalizing audio (max val: {max_abs_val:.2f})...")
                    audio_numpy = audio_numpy / max_abs_val
                elif max_abs_val == 0: raise ValueError("Audio appears to be silent.")
                result = asr_pipeline({"sampling_rate": sample_rate, "raw": audio_numpy})
            else:
                raise TypeError(f"Unexpected audio input format: {type(audio_input)}. Expected numpy array.")

            end_time = time.time()
            print(f"Transcription task finished in {end_time - start_time:.2f} seconds.")

            # Extract text result
            final_transcription = result.get('text', '').strip() if isinstance(result, dict) else str(result).strip()
            if not final_transcription:
                final_transcription = "(No speech detected or model returned empty transcription)"
                print("Result: No speech detected.")
            else:
                 print(f"Result Preview: {final_transcription[:100]}...")

            # --- Log Capture Ends Here ---

        # 4. Final Success Status (Append remaining logs and update UI)
        # Get only the logs generated *since the last update*
        current_logs = log_stream.getvalue()
        logs_from_transcription = current_logs[len(logs_after_loading):]
        log_accumulator.append(logs_from_transcription)
        logs = "".join(log_accumulator) # Complete logs

        status_updates[status_output] = gr.update(value="✅ Transcription complete!")
        status_updates[output_textbox] = gr.update(value=final_transcription)
        status_updates[log_output] = gr.update(value=logs) # Update log display with ALL logs
        yield status_updates

    except Exception as e:
        # --- Log Capture Ends Here on Error ---
        print(f"ERROR encountered during process: {e}") # Log the error itself
        logs = log_stream.getvalue() # Get everything captured up to the error
        logs += f"\n\n--- ERROR ENCOUNTERED ---\n{type(e).__name__}: {e}" # Append error info

        # Determine user-friendly error message
        if isinstance(e, FileNotFoundError): error_message = f"Error: {e}"
        elif isinstance(e, ValueError): error_message = f"Input Error: {e}"
        elif "out of memory" in str(e).lower(): error_message = "Error: Out of memory."
        elif "ffmpeg" in str(e).lower(): error_message = "Error: FFmpeg issue or audio format."
        elif isinstance(e, RuntimeError) and "Failed to load model" in str(e): error_message = str(e)
        else: error_message = f"An unexpected error occurred. Check logs. Error: {str(e)}"

        # 5. Final Error Status
        status_updates[status_output] = gr.update(value=f"❌ Error! See logs.")
        status_updates[output_textbox] = gr.update(value=error_message)
        status_updates[log_output] = gr.update(value=logs) # Display logs including error
        yield status_updates
    finally:
         log_stream.close()


# --- Gradio UI Definition ---

# Customize theme further for a professional look
theme = themes.Soft(
    font=[themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue=themes.colors.blue,
    secondary_hue=themes.colors.sky,
    radius_size="sm",  # Use string literal "sm"
    spacing_size="md", # Use string literal "md"
)

# Optional CSS for minor adjustments
css = """
.gradio-container { padding-top: 20px !important; padding-bottom: 20px !important; }
footer { display: none !important; } /* Hide default Gradio footer for cleaner look */
.stt-footer { text-align: center; color: #888; font-size: 0.9em; margin-top: 20px; }
"""

with gr.Blocks(theme=theme, title="Speech-to-Text Pro Demo", css=css) as demo:
    gr.Markdown(
        """
        # 🎙️ Advanced Speech-to-Text Demo
        Select a model, provide audio, and get accurate transcriptions with live status and detailed logs.
        """
    )

    with gr.Row(equal_height=False):
        # --- Left Column: Inputs & Control ---
        with gr.Column(scale=1, min_width=380):
            with gr.Group(): # Use gr.Group for visual grouping
                gr.Markdown("### 1. Configuration")
                model_selector = gr.Dropdown(
                    label="Select ASR Model",
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],
                    interactive=True,
                    elem_id="model_dropdown" # Optional element ID
                )
                gr.Markdown("### 2. Audio Input")
                audio_input = gr.Audio(
                    label="Upload File or Record via Microphone",
                    sources=["upload", "microphone"],
                    type="numpy", # Handles both sources, provides array & sr
                    elem_id="audio_input"
                )
                gr.Markdown("### 3. Transcribe")
                transcribe_button = gr.Button("Transcribe Audio", variant="primary", elem_id="transcribe_button")


        # --- Right Column: Outputs & Logs ---
        with gr.Column(scale=2, min_width=500):
             with gr.Group(): # Use gr.Group for visual grouping
                gr.Markdown("### Results & Status")
                output_textbox = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=8, # Increased lines slightly
                    interactive=False,
                    show_copy_button=True,
                    elem_id="transcription_output"
                )
                status_output = gr.Textbox(
                    label="Current Status",
                    value="Idle. Select model and provide audio.",
                    lines=1,
                    interactive=False,
                    elem_id="status_output"
                )
                # Use Accordion for Logs - collapsed by default for cleaner initial view
                with gr.Accordion("📄 View Detailed Logs", open=False):
                    log_output = gr.Textbox(
                        label="Process Logs",
                        placeholder="Logs from the model loading and transcription process will appear here after completion...",
                        lines=12, # Increased lines
                        max_lines=25,
                        interactive=False,
                        show_copy_button=True,
                        elem_id="log_output"
                    )

    # Custom Footer
    gr.Markdown(
        f"""
        ---
        <p class="stt-footer">
        App running locally | Device: {device.upper()} | Current Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
        </p>
        """,
        elem_id="custom_footer"
    )


    # --- Connect UI Components to Generator Function ---
    transcribe_button.click(
        fn=transcribe_audio,
        inputs=[model_selector, audio_input],
        # List ALL components that the function yields updates for
        outputs=[output_textbox, status_output, log_output]
        # Progress bar is automatically handled for generators by Gradio
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        share=False, # Keep it local unless needed
        # server_name="0.0.0.0" # Uncomment to allow access from other devices on your network
        )
