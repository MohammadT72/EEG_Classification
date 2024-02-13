import streamlit as st
import pandas as pd
from torcheeg import transforms
from torcheeg.utils import plot_signal,plot_raw_topomap,plot_feature_topomap
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT,DEAP_CHANNEL_LIST
import numpy as np
import mne


map_transform=transforms.Compose([
    transforms.ToTensor(),
])

def draw_shapley_plot():
    names = ["Cz", "Fz", "Fp1", "F7", "F3"]  # Add all names
    # Generate some fake data
    values = np.random.rand(len(names))

    # Create a DataFrame
    df = pd.DataFrame(list(zip(names, values)), columns=['Name', 'Value'])

    # Sort the DataFrame by value
    df = df.sort_values('Value')

    # Create the plot in Streamlit
    st.bar_chart(df.set_index('Name'))

def plot_signal_fn(epochs, selected_channels, sampling_rate=128):
    eeg = map_transform(eeg=epochs.get_data())['eeg'][0]
    img = plot_signal(eeg,
                  channel_list=selected_channels,
                  sampling_rate=sampling_rate)
    st.markdown('----------------')
    st.write('Plotting Two second of the raw signal')
    st.image(img)

def plot_raw_topomap_fn(epochs,selected_channels, sampling_rate=128):
    eeg = map_transform(eeg=epochs.get_data())['eeg'][0]
    img = plot_raw_topomap(eeg,
                 channel_list=selected_channels,
                 sampling_rate=sampling_rate)
    st.markdown('----------------')
    st.write('Plotting Two second Topomap')
    st.image(img)
    
def plot_feature_topomap_fn(epochs, selected_channels,):
    eeg = map_transform(eeg=epochs.get_data())['eeg'][0]
    img = plot_feature_topomap(eeg[:,:4],
                 channel_list=selected_channels,
                 feature_list=["theta", "alpha", "beta", "gamma"])
    st.markdown('----------------')
    st.write('Plotting Feature Topomap')
    st.image(img)

def plot_handler(names:list, info_dict, **kwargs):
    start_time=kwargs['start']
    end_time=kwargs['end']
    raw=info_dict['raw_data']
    selected_channels=kwargs['selected_channels']
    sampling_rate=info_dict['sampling_rate']
    duration=end_time-start_time
    if duration==0:
        raw.crop(tmin=start_time, tmax=start_time+1)
        duration=1
    else:
        raw.crop(tmin=start_time, tmax=end_time)

    epochs = mne.make_fixed_length_epochs(raw, duration=duration)
    epochs_copy = epochs.copy()
    epochs_copy.load_data().pick_channels(selected_channels)
    for name in names:
        if name=='Raw Signal':
            plot_signal_fn(epochs_copy, selected_channels,sampling_rate)
        if name=='Raw Topomap':
            plot_raw_topomap_fn(epochs_copy, selected_channels,sampling_rate)
        if name=='Feature Topomap':
            plot_feature_topomap_fn(epochs_copy, selected_channels)
