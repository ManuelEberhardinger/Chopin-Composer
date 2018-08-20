from midihelper import MidiHelper, generate_midi_from_prediction
from neuralnetwork import CustomModel

midiHelper = MidiHelper("chopin-midi", "chopinNotesWithOffset")
network_input, network_output = midiHelper.generate_sequences(100)

net = CustomModel(network_input, network_output, midiHelper.get_notes())
net.train(epochs=200, batch_size=64)
generated_music = net.generate_music(500)
generate_midi_from_prediction(generated_music, "test_65_epochs_offset")
