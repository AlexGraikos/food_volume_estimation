import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


class TrainingHistoryVisualizer():
    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        """Parse command line arguments.
        
        Returns:
            args: Command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description='Visualize training.')
        parser.add_argument('--training_history', type=str,
                            help='Training history file path.',
                            metavar='/path/to/history',
                            required=True)
        parser.add_argument('--single_plot', action='store_true',
                            help='Plot all results in a single figure.',
                            default=False)
        args = parser.parse_args()
        return args

    def visualize_history(self, training_history_file, single_plot=False):
        """Plot training history variables.

        Inputs:
            training_history_file: JSON file containing training history
                variables.
            single_plot: Flag to condense all plots in a single figure.
        """
        # Load training history from JSON and define values to plot
        with open(training_history_file, 'r') as read_file:
            training_history_dict = json.load(read_file)
        variables_to_plot = [
            'loss', 'scale1_reprojections_loss', 'scale2_reprojections_loss', 
            'scale3_reprojections_loss', 'scale4_reprojections_loss',
            'depth_net_loss', 'lr']
        print('[*] Loaded training history from', training_history_file)

        if single_plot:
            # Print variables in a single figure on a 3x3 grid
            fig = plt.figure()
            spec = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

            # Print the total loss in a single row
            key = variables_to_plot[0]
            vals = training_history_dict[key]
            fig.add_subplot(spec[0,:])
            plt.title(key)
            plt.plot(range(1, len(vals)+1), vals)
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Integer x labels

            # Fill the grid with the rest of the variables
            r = 1
            c = 0
            for key in variables_to_plot[1:]:
                vals = training_history_dict[key]
                fig.add_subplot(spec[r,c])
                plt.title(key)
                plt.plot(range(1, len(vals)+1), vals)
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                # Update column and row indices
                c = (c + 1) % 3
                r = (r + 1) if (c == 0) else r

        else:
            # Print each variable in a separate figure
            for key in variables_to_plot:
                plt.figure()
                vals = training_history_dict[key]
                plt.title(key)
                plt.plot(range(1, len(vals)+1), vals)
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

if __name__ == '__main__':
    visualizer = TrainingHistoryVisualizer()
    visualizer.visualize_history(visualizer.args.training_history,
                                 visualizer.args.single_plot)

