from midihelper import MidiHelper, generate_midi_from_prediction
from neuralnetwork import CustomModel
import argparse

parser = argparse.ArgumentParser(description='Generate classical music in the style of Frederic Chopin.')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('-m', '--midifile', nargs='?', dest="midifile", type=str, help='output file, in midi format')
parser.add_argument("modelName", help="the name of the model which should be created, must match the method name ",
                    type=str)
args = parser.parse_args()

midiHelper = MidiHelper("chopin-midi", "chopinNotesWithOffset")
network_input, network_output = midiHelper.generate_sequences(100)

net = CustomModel(network_input, network_output, midiHelper.get_notes(), args.modelName, args.train)
net.train_model(epochs=200, batch_size=64)
generated_music = net.generate_music(1000)
midi_name = args.midifile if args.midifile is not None else 'default.mid'
midi_name = args.modelName if args.midifile is None and args.train is True else midi_name
generate_midi_from_prediction(generated_music, midi_name)

