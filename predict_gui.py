import tkinter as tk
from tkinter import filedialog, messagebox
import pretty_midi
import numpy as np
import joblib

model = joblib.load("../model.pkl")

def extract_features(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
                break
        if not notes:
            return None
        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        intervals = np.diff(sorted([note.start for note in notes]))

        return np.array([
            np.mean(pitches), np.var(pitches),
            np.mean(durations), np.var(durations),
            len(notes),
            np.mean(intervals) if len(intervals) > 0 else 0
        ]).reshape(1, -1)
    except:
        return None

def predict():
    file_path = filedialog.askopenfilename(filetypes=[("MIDI Files","*.mid")])
    if not file_path:
        return

    features = extract_features(file_path)
    if features is None:
        messagebox.showerror("Error", "Invalid or unsupported MIDI file.")
        return

    prediction = model.predict(features)[0]
    result = "🎵 HUMAN-COMPOSED MUSIC" if prediction == 0 else "🤖 AI-GENERATED MUSIC"
    messagebox.showinfo("Prediction Result", result)

root = tk.Tk()
root.title("AI-Human Music Classifier")
tk.Button(root, text="Select MIDI File & Predict", command=predict).pack(padx=20, pady=20)
root.mainloop()
