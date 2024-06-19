import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import json
import pandas as pd
import requests
from io import BytesIO
import base64
import time

# RETSPL values for the specified frequencies
retspl_values = {
    125: 30.5, 250: 18, 500: 11, 750: 6, 1000: 5.5,
    1500: 5.5, 2000: 4.5, 3000: 2.5, 4000: 9.5, 6000: 17,
    8000: 17.5
}

# GitHub configuration
GITHUB_TOKEN = "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"
GIST_ID = "5f90c089b2aff7b81eff824177d745bf"
GIST_FILENAME = "calibration_data"

def preload_audio_files(progress_callback):
    audio_files = {}
    total_files = len(retspl_values)
    loaded_files = 0

    for freq in retspl_values.keys():
        url = f"https://raw.githubusercontent.com/NGU246/HearMetrics/main/static/tone_{freq}Hz.wav"
        response = requests.get(url)
        if response.status_code == 200:
            audio_data, samplerate = sf.read(BytesIO(response.content))
            temp_wav_path = f"temp_{freq}.wav"
            sf.write(temp_wav_path, audio_data, samplerate)
            with open(temp_wav_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                audio_files[freq] = audio_base64
        else:
            st.warning(f"Could not load audio file for {freq} Hz. Status code: {response.status_code}")

        loaded_files += 1
        progress_callback(loaded_files / total_files)

    return audio_files

def load_calibration_data():
    url = f'https://api.github.com/gists/{GIST_ID}'
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        gist_data = response.json()
        file_content = gist_data['files'][f'{GIST_FILENAME}.json']['content']
        return json.loads(file_content)
    else:
        return {str(freq): {'left': 0, 'right': 0} for freq in retspl_values.keys()}

def initialize_session_state(progress_callback):
    total_steps = 3
    current_step = 0

    if 'calibration' not in st.session_state:
        st.session_state.calibration = load_calibration_data()
    current_step += 1
    progress_callback(current_step / total_steps)

    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = preload_audio_files(progress_callback)
    current_step += 1
    progress_callback(current_step / total_steps)

    defaults = {
        'playing': False,
        'current_frequency': '125',
        'current_volume_db_hl': 45,
        'current_calibration_db': 0,
        'ear': 'left',
        'tone_type': 'steady',
        'thresholds': {'left': {}, 'right': {}},
        'analyzer_ear': 'left',
        'play_button_pressed': False,
        'start_time': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    current_step += 1
    progress_callback(current_step / total_steps)

def save_calibration_data():
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    url = f'https://api.github.com/gists/{GIST_ID}'
    data = {
        'files': {
            f'{GIST_FILENAME}.json': {
                'content': json.dumps(st.session_state.calibration, indent=2)
            }
        }
    }
    response = requests.patch(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        st.success("Calibration data saved successfully to GitHub Gist.")
    else:
        st.error(f"Failed to save calibration data: {response.status_code}")

def dbhl_to_volume(db_hl, retspl, calibration):
    volume = 10 ** ((db_hl - retspl + calibration) / 20)
    return volume

def adjust_volume(audio_data, volume):
    return audio_data * volume

def play_sound(frequency, volume, ear):
    st.write(f"Playing frequency: {frequency} Hz")
    st.write(f"Volume: {volume}")
    volume = max(0, min(volume, 1))

    if frequency not in st.session_state.audio_files:
        st.error(f"Audio file for {frequency} Hz is not loaded.")
        return

    audio_base64 = st.session_state.audio_files[frequency]
    audio_bytes = base64.b64decode(audio_base64)
    audio_data, samplerate = sf.read(BytesIO(audio_bytes))
    adjusted_audio_data = adjust_volume(audio_data, volume)

    if ear == 'left':
        adjusted_audio_data = np.array([adjusted_audio_data, np.zeros_like(adjusted_audio_data)]).T
    elif ear == 'right':
        adjusted_audio_data = np.array([np.zeros_like(adjusted_audio_data), adjusted_audio_data]).T

    buffer = BytesIO()
    sf.write(buffer, adjusted_audio_data, samplerate, format='WAV')
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode()
    audio_html = f"""
        <audio id="audio" controls>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

    if st.session_state.play_button_pressed:
        st.markdown(
            """
            <script>
                document.getElementById('audio').play();
            </script>
            """,
            unsafe_allow_html=True,
        )

def stop_sound():
    st.write("Stopping audio.")
    st.markdown(
        """
        <script>
            var audio = document.getElementById('audio');
            if (audio) {
                audio.pause();
                audio.currentTime = 0;
            }
        </script>
        """,
        unsafe_allow_html=True,
    )

def calibration_menu():
    st.header("Calibration")
    frequencies = list(retspl_values.keys())
    current_frequency = st.selectbox("Select Frequency for Calibration (Hz)", frequencies, index=frequencies.index(int(st.session_state.current_frequency)), key="frequency_select")
    st.session_state.current_frequency = str(current_frequency)
    
    ear = st.selectbox("Select Ear for Calibration", ['left', 'right'], key="ear_select_calibration")
    st.session_state.ear = ear

    volume_db_hl = st.number_input("Calibration Volume (dB HL)", value=st.session_state.current_volume_db_hl, min_value=0, max_value=90, step=1, key="volume_input_calibration")
    st.session_state.current_volume_db_hl = volume_db_hl

    retspl = retspl_values[int(current_frequency)]
    calibration = st.session_state.calibration[str(current_frequency)][ear]
    st.session_state.current_calibration_db = calibration

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("+1 dB", key="calibration_up_1"):
        st.session_state.current_calibration_db += 1
    if col2.button("-1 dB", key="calibration_down_1"):
        st.session_state.current_calibration_db -= 1
    if col3.button("+5 dB", key="calibration_up_5"):
        st.session_state.current_calibration_db += 5
    if col4.button("-5 dB", key="calibration_down_5"):
        st.session_state.current_calibration_db -= 5

    st.session_state.current_calibration_db = max(min(st.session_state.current_calibration_db, 100), -100)
    st.session_state.calibration[str(current_frequency)][ear] = st.session_state.current_calibration_db

    volume = dbhl_to_volume(volume_db_hl, retspl, st.session_state.current_calibration_db)

    if st.button("Play Tone", key="play_tone_button_calibration"):
        st.session_state.play_button_pressed = True
        st.session_state.start_time = time.time()
        play_sound(int(current_frequency), volume, st.session_state.ear)
    
    if st.button("Stop Tone", key="stop_tone_button_calibration"):
        stop_sound()
        st.session_state.play_button_pressed = False
        st.session_state.start_time = None

    if st.session_state.play_button_pressed:
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

    if st.button("Set Calibration for Current Frequency", key="set_calibration_button"):
        st.session_state.calibration[str(current_frequency)][ear] = st.session_state.current_calibration_db
        save_calibration_data()
        st.write(f"Calibration for {current_frequency} Hz ({ear}) set to {st.session_state.current_calibration_db} dB")

def plot_audiogram():
    frequencies_list = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    freq_indices = list(range(len(frequencies_list)))
    left_thresholds = [st.session_state.thresholds['left'].get(str(freq), None) for freq in frequencies_list]
    right_thresholds = [st.session_state.thresholds['right'].get(str(freq), None) for freq in frequencies_list]

    fig, ax = plt.subplots()
    if any(left_thresholds):
        ax.plot(freq_indices, left_thresholds, marker='x', linestyle='-', label='Left Ear (Blue)', color='blue')
    if any(right_thresholds):
        ax.plot(freq_indices, right_thresholds, marker='o', linestyle='-', label='Right Ear (Red)', color='red')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Hearing Threshold (dB HL)')
    ax.set_title('Audiogram')
    ax.set_xticks(freq_indices)
    ax.set_xticklabels(['125', '250', '500', '750', '1k', '1.5k', '2k', '3k', '4k', '6k', '8k'])
    ax.set_ylim(90, 0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    return fig

def categorize_hearing_loss(thresholds):
    def get_severity(threshold):
        if threshold <= 25:
            return "Normal Hearing"
        elif 26 <= threshold <= 40:
            return "Mild Hearing Loss"
        elif 41 <= 55:
            return "Moderate Hearing Loss"
        elif 56 <= 70:
            return "Moderately Severe Hearing Loss"
        elif 71 <= 90:
            return "Severe Hearing Loss"
        else:
            return "Profound Hearing Loss"
    
    low_freqs = [250, 500]
    mid_freqs = [1000, 2000]
    high_freqs = [4000, 8000]

    low_loss = max((thresholds.get(str(f), -1) for f in low_freqs), default=-1)
    mid_loss = max((thresholds.get(str(f), -1) for f in mid_freqs), default=-1)
    high_loss = max((thresholds.get(str(f), -1) for f in high_freqs), default=-1)

    low_severity = get_severity(low_loss) if low_loss != -1 else "No Data"
    mid_severity = get_severity(mid_loss) if mid_loss != -1 else "No Data"
    high_severity = get_severity(high_loss) if high_loss != -1 else "No Data"

    results = []
    
    if low_severity != "Normal Hearing" and low_severity != "No Data":
        results.append(f"Low Frequency (250-500 Hz): {low_severity}")
    if mid_severity != "Normal Hearing" and mid_severity != "No Data":
        results.append(f"Mid Frequency (1000-2000 Hz): {mid_severity}")
    if high_severity != "Normal Hearing" and high_severity != "No Data":
        results.append(f"High Frequency (4000-8000 Hz): {high_severity}")
    
    if not results:
        return "Normal Hearing across all frequencies"
    
    return " | ".join(results)

if 'analyzer_ear' not in st.session_state:
    st.session_state.analyzer_ear = 'left'

def hearing_loss_analyzer():
    ear = st.session_state.analyzer_ear
    thresholds = st.session_state.thresholds[ear]
    if thresholds:
        hearing_loss_category = categorize_hearing_loss(thresholds)
        st.write(f"Hearing loss for the {ear} ear: {hearing_loss_category}")
    else:
        st.write("No thresholds recorded for the selected ear.")

def main_menu():
    st.title("HearMetrics ProScreener")

    frequencies = list(map(str, [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]))
    current_frequency = st.session_state.current_frequency
    current_volume_db_hl = st.session_state.current_volume_db_hl

    st.markdown("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        if st.button("←", key="freq_down"):
            prev_index = (frequencies.index(current_frequency) - 1) % len(frequencies)
            st.session_state.current_frequency = frequencies[prev_index]
            st.rerun()
    with col3:
        st.write(f"**Frequency:** {st.session_state.current_frequency} Hz")
        retspl = retspl_values[int(st.session_state.current_frequency)]
        calibration = st.session_state.calibration[st.session_state.current_frequency][st.session_state.ear]
        volume = dbhl_to_volume(st.session_state.current_volume_db_hl, retspl, calibration)
        if st.button("Play Tone", key="play_tone_button_main"):
            st.session_state.play_button_pressed = True
            st.session_state.start_time = time.time()
            play_sound(int(st.session_state.current_frequency), volume, st.session_state.ear)
    with col5:
        if st.button("→", key="freq_up"):
            next_index = (frequencies.index(current_frequency) + 1) % len(frequencies)
            st.session_state.current_frequency = frequencies[next_index]
            st.rerun()

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        if st.button("-5 dB", key="vol_down"):
            st.session_state.current_volume_db_hl = max(current_volume_db_hl - 5, 0)
            st.rerun()
    with col3:
        st.write(f"**Volume:** {st.session_state.current_volume_db_hl} dB HL")
        if st.button("Record Threshold", key="record_threshold"):
            st.session_state.thresholds[st.session_state.ear][st.session_state.current_frequency] = st.session_state.current_volume_db_hl
            st.write(f"Recorded threshold for {st.session_state.current_frequency} Hz ({st.session_state.ear}): {st.session_state.current_volume_db_hl} dB HL")
    with col5:
        if st.button("+5 dB", key="vol_up"):
            st.session_state.current_volume_db_hl = min(current_volume_db_hl + 5, 90)
            st.rerun()

    ear = st.radio("Select Ear", ['left', 'right'], index=0 if st.session_state.ear == 'left' else 1, key="ear_radio_main")
    st.session_state.ear = ear

    tone_type = st.radio("Select Tone Type", ['steady', 'pulsed'], index=0 if st.session_state.tone_type == 'steady' else 1, key="tone_type_radio")
    st.session_state.tone_type = tone_type

    retspl = retspl_values[int(st.session_state.current_frequency)]
    calibration = st.session_state.calibration.get(str(st.session_state.current_frequency), {}).get(ear, 0)
    volume = dbhl_to_volume(st.session_state.current_volume_db_hl, retspl, calibration)

    display_option = st.selectbox("Display Option", ["Table", "Audiogram"], key="display_option")

    if display_option == "Table":
        if st.session_state.thresholds:
            thresholds_data = {'Frequency (Hz)': [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]}
            thresholds_data['Left Ear (dB HL)'] = [st.session_state.thresholds['left'].get(str(freq), '') for freq in thresholds_data['Frequency (Hz)']]
            thresholds_data['Right Ear (dB HL)'] = [st.session_state.thresholds['right'].get(str(freq), '') for freq in thresholds_data['Frequency (Hz)']]
            df_thresholds = pd.DataFrame(thresholds_data)
            st.table(df_thresholds)
    elif display_option == "Audiogram":
        audiogram_placeholder = st.empty()
        audiogram_placeholder.pyplot(plot_audiogram())

    ear = st.radio("Select Ear for Analyzer", ['left', 'right'], index=0 if st.session_state.ear == 'left' else 1, key="ear_radio_analyzer")
    st.session_state.analyzer_ear = ear

    if st.button("Hearing Loss Analyzer", key="hearing_loss_analyzer"):
        hearing_loss_analyzer()

    if st.button("Stop Tone", key="stop_tone_button_main"):
        stop_sound()
        st.session_state.play_button_pressed = False
        st.session_state.start_time = None

    if st.session_state.play_button_pressed:
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

def show_loading_page():
    st.title("Loading HearMetrics ProScreener")
    progress_bar = st.progress(0)

    def progress_callback(progress):
        progress_bar.progress(int(progress * 100))

    initialize_session_state(progress_callback)
    st.experimental_rerun()

if 'calibration' not in st.session_state or 'audio_files' not in st.session_state:
    show_loading_page()
else:
    option = st.sidebar.selectbox("Go to", ["Main Menu", "Calibration"], key="navigation")
    if option == "Main Menu":
        main_menu()
    elif option == "Calibration":
        calibration_menu()
