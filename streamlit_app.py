import streamlit as st
import pandas as pd
import numpy as np
import mne
from torcheeg.datasets import CSVFolderDataset

from stream_plots import plot_handler
from stream_model import MyPredictor
import os

# Load the EEG dataset
@st.cache
def load_data(file_path):
    eeg_data = load_eeg(file_path)
    return eeg_data



def default_read_fn(file_path, **kwargs):
        # Load EEG file
        raw = mne.io.read_raw(file_path)
        # Convert raw to epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=1)
        # Return EEG data
        return epochs, raw

def base_analysis(uploaded_file):
    # Load the raw data
    raw = mne.io.read_raw_fif(uploaded_file, preload=True)

    # Get the names of the channels
    channel_names = raw.info['ch_names']

    # Calculate the duration of the EEG recording and round it up
    duration = int(np.ceil(raw.times[-1]))

    # Get the sampling rate
    sampling_rate = raw.info['sfreq']

    # Return a dictionary with the data
    return {
        'channel_names': channel_names,
        'duration': duration,
        'sampling_rate': sampling_rate,
        'raw_data': raw,
    }

def select_range(duration):
    # If the recording is less than two seconds long, select the entire recording
    if duration < 2:
        start_time = 0
        end_time = duration
    else:
        # Otherwise, select a two-second range
        start_time = 0
        end_time = 2
    return int(start_time),int(end_time)
def select_channels(info_dict):
    # List of channel names
    channels = info_dict['channel_names']

    # Create a selection box for the channels
    selected_channels = st.multiselect('Select the channels you want to plot', channels, default=channels)

    return selected_channels
def select_plots():
    # List of channel names
    plots = ['Raw Signal','Raw Topomap', 'Feature Topomap']

    # Create a selection box for the channels
    selected_plot = st.multiselect('Select the type of plot you want', plots, default='Raw Signal')

    return selected_plot
def select_test_files():
    # List of channel names
    files = ['Rest','Stress']

    # Create a selection box for the channels
    selected_file = st.selectbox('Select test a test file', files)

    return selected_file

def add_info():
    # Create a dictionary to store user information
    user_info = {}
    
    # Create form
    with st.expander('patient info'):
        with st.form(key='user_profile_form'):
            st.write('Please fill out the form:')
            
            user_info['person_relation'] = st.text_input('Person Relation')
            user_info['person_income'] = st.number_input('Person Income', step=1)
            user_info['person_employment'] = st.text_input('Person Employment')
            user_info['person_education'] = st.text_input('Person Education')
            user_info['person_family'] = st.text_input('Person Family')
            user_info['underlying_conditions'] = st.text_input('Underlying Conditions')
            user_info['substance_use'] = st.text_input('Substance Use')
            user_info['mental_disorders'] = st.text_input('Mental Disorders')
            user_info['neurological_disorders'] = st.text_input('Neurological Disorders')
            user_info['date'] = st.date_input('Date')
            user_info['type'] = st.text_input('Type')
            user_info['phq9'] = st.number_input('PHQ-9 Score', step=1)
            user_info['bai'] = st.number_input('BAI Score', step=1)
            user_info['stai state'] = st.number_input('STAI State Score', step=1)
            user_info['stai trait'] = st.number_input('STAI Trait Score', step=1)
            user_info['voci'] = st.number_input('VOCI Score', step=1)
            user_info['atqb'] = st.number_input('ATQB Score', step=1)
            user_info['sds'] = st.number_input('SDS Score', step=1)
            user_info['hash'] = st.text_input('Hash')
            user_info['eyes'] = st.text_input('Eyes')
            user_info['file_score'] = st.number_input('File Score', step=1)
            user_info['cls_score'] = st.number_input('CLS Score', step=1)
            user_info['pop_score'] = st.number_input('POP Score', step=1)
            
            # When the user presses the 'Submit' button, print the user_info dictionary
            if st.form_submit_button('Submit'):
                st.write(user_info)



mypredictor=MyPredictor()
mypredictor.load_model()

def main():
    st.title("EEG AI MVP")
    st.write('Please provide a FIF, BDF, or EDF file')
    uploaded_file = st.file_uploader("Choose a fif file", type="fif")
    selected_file = select_test_files()
    # Create a simple dataframe
    data = {
            'Patient ID': ['DX12565'],
            'Name': ['John'],
            'Age': [28],
            'Gender':['Male']}
    df = pd.DataFrame(data)
    st.table(df)
    st.button('Add Patient information', on_click=add_info)
    if uploaded_file==None:
        if selected_file=='Rest':
            uploaded_file='./1_2_eeg.fif'
        elif selected_file=='Stress':
            uploaded_file='1_2_eeg.fif'
    st.title("Basic report")
    info_dict = base_analysis(uploaded_file)
    selected_channels = select_channels(info_dict)
    selected_plots = select_plots()
    selected_rage = select_range(info_dict['duration'])
    # file_path = st.text_input("Enter the path to your EEG dataset:")
    start,end = st.slider('Select a range time', 0, info_dict['duration'], selected_rage)
    plot_handler(names=selected_plots, info_dict=info_dict,
                 start=start,end=end, selected_channels=selected_channels, 
                 selected_rage=selected_rage)
    preds, conf = mypredictor.predict(info_dict['raw_data'], start, end)
    st.title('AI analysis results: ')
    if preds == 'Stress':
        st.write('The patient had Stress on session')
        st.write(f'Prediction confidence: {conf:.2f}')
if __name__ == "__main__":
    main()
