import torch
import numpy as np
from mido import Message, MidiFile, MidiTrack

##### GENERATING MUSICAL SEQUENCES #####
# Now I want to generate a musical sequence using the trained model.

# Define a function that, given a start sequence, generates a sequence of N notes:
def generate_notes(model, rnn, start_sequence, num_notes_to_generate, threshold):
    model.eval()  # Set model to evaluation mode

    # Initialize the sequence and hidden state
    generated_sequence = start_sequence.clone()  # Shape: (1, sequence_length, input_size)
    hidden = None

    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_notes_to_generate):
            # Get the hidden states for the current sequence
            _, hidden = rnn(generated_sequence, hidden)
            
            # Use the hidden states to predict the next note
            output = model(hidden[:, -1, :])  # Use the last hidden state
            
            # Threshold or sample the prediction
            next_note = (output >= threshold).float().unsqueeze(1)  # Shape: (1, 1, input_size)
            
            # Append the next note to the sequence
            generated_sequence = torch.cat((generated_sequence, next_note), dim=1)

    return generated_sequence

# Load starting sequence:
start_sequence = torch.rand((1, 1000, 12))

# Load saved model:
rnn = torch.load("rnn.pth")
readout = torch.load("readout.pth")


# Generate a sequence of 100 notes:
generated_sequence = generate_notes(readout,
                                    rnn,
                                    start_sequence,
                                    num_notes_to_generate=100,
                                    threshold=0.5)


def sequence_to_notes_and_chords(sequence):
    # Chromatic scale mapping
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    result = []
    
    for time_step in sequence:
        # Find active notes (nonzero indices)
        active_indices = np.where(time_step > 0)[0]
        if len(active_indices) == 0:
            result.append("Rest")  # No active note
        elif len(active_indices) == 1:
            # Single note
            result.append(chromatic_scale[active_indices[0]])
        else:
            # Chord
            chord = "+".join([chromatic_scale[i] for i in active_indices])
            result.append(chord)
    
    return result

from mido import Message, MidiFile, MidiTrack

def sequence_to_midi(sequence, output_file="output.mid"):
    """
    Converts a sequence of notes into a MIDI file.

    Args:
        sequence (list): A sequence where each step is a list of 12 binary values
                         (1 for active note, 0 for inactive).
        output_file (str): Name of the output MIDI file.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Base note number for the chromatic scale (C4 = 60)
    base_note = 60
    
    for time_step in sequence:
        active_notes = [i for i, v in enumerate(time_step) if v > 0]
        
        # Turn on active notes
        for note in active_notes:
            track.append(Message('note_on', note=base_note + note, velocity=64, time=0))
        
        # Turn off all notes at the end of the time step
        for note in active_notes:
            track.append(Message('note_off', note=base_note + note, velocity=64, time=480))
    
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")

# Example sequence (12 notes in chromatic scale)
sequence = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C#+D#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Rest
]

# Generate and save the MIDI file
sequence_to_midi(sequence, "output.mid")

import pygame

def play_midi(file):
    """
    Plays a MIDI file using pygame.
    
    Args:
        file (str): Path to the MIDI file.
    """
    pygame.init()
    pygame.mixer.init()
    
    try:
        pygame.mixer.music.load(file)
        print(f"Playing {file}...")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass  # Wait until the music is done playing
    except pygame.error as e:
        print(f"Error playing MIDI file: {e}")
    finally:
        pygame.quit()

# Play the generated MIDI file
play_midi("output.mid")