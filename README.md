# AI Music Classifier (Human vs AI-Generated Music)

A machine learning system that classifies music as human-composed or AI-generated using MIDI file features.  
The model extracts symbolic features (notes, tempo, duration, pitch) and uses them to train a classifier.

## Features
- Classifies MIDI files into Human or AI-generated
- Feature extraction from symbolic music (MIDI)
- Confusion matrix and classification metrics
- GUI for testing custom MIDI files
- Lightweight and fast processing

## Technologies Used
- Python
- Scikit-Learn
- PrettyMIDI
- NumPy, Pandas
- Tkinter (GUI)

## Dataset
A dataset of **80 MIDI files**:
- 40 Human-composed
- 40 AI-generated
A small sample dataset is included in `data.zip`

## Results
<img width="215" height="94" alt="{C2F41569-9FB2-4BF8-BD88-1BA8FD2CB236}" src="https://github.com/user-attachments/assets/e26be8f4-024c-4f09-980e-5e1a1d47b018" />

### Confusion Matrix
<img width="267" height="126" alt="{DDDD83AE-35F3-4193-9264-3B7CEF739426}" src="https://github.com/user-attachments/assets/72731f2e-fa36-4425-8588-cdc288186f67" />

### Install Requirements
pip install numpy pandas scikit-learn pretty_midi tkinter
