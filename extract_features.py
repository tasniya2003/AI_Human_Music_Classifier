import pretty_midi
import numpy as np
import os
import pandas as pd

def extract_features(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note)
                break  

        if not notes:
            return None

        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        intervals = np.diff(sorted([note.start for note in notes]))

        return {
            "pitch_mean": np.mean(pitches),
            "pitch_var": np.var(pitches),
            "dur_mean": np.mean(durations),
            "dur_var": np.var(durations),
            "note_count": len(notes),
            "int_mean": np.mean(intervals) if len(intervals) > 0 else 0
        }
    except:
        return None


def process_folder(folder_path, label):
    rows = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid"):
            features = extract_features(os.path.join(folder_path, file))
            if features:
                features["label"] = label
                rows.append(features)
    return rows


human = process_folder("data/human", 0)
ai = process_folder("data/ai_generated", 1)

df = pd.DataFrame(human + ai)
df.to_csv("../dataset.csv", index=False)
print("Feature extraction completed successfully!")