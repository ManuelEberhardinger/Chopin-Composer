from midihelper import MidiHelper, generate_midi_from_prediction
from neuralnetwork import CustomModel
import argparse

parser = argparse.ArgumentParser(description='Generate classical music in the style of Frederic Chopin.')
parser.add_argument("modelName", help="the name of the model which should be created",
                    type=int)
parser.add_argument("train", help="if true the model should be trained, if false music should be generated")
parser.add_argument("midiFile", help="the name of the generated midi file")
args = parser.parse_args()

midiHelper = MidiHelper("chopin-midi", "chopinNotesWithOffset")
network_input, network_output = midiHelper.generate_sequences(100)

net = CustomModel(network_input, network_output, midiHelper.get_notes(), args.modelName, train=args.train)
net.train(epochs=200, batch_size=64)
generated_music = net.generate_music(1000)
generate_midi_from_prediction(generated_music, args.midiFile)

