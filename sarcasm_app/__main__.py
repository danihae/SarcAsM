# Usage of this software for commercial purposes without a license is strictly prohibited.

from sarcasm_app import Application


def main():
    application = Application()
    application.init_gui()


if __name__ == '__main__':
    main()

"""
TODO:
In general, please check whether the default values of all methods are the same as in the package.
Can you somehow automate that using the inspect package? Would that make sense?
---> does not make sense: instead update the current default values in /sarcasm_app/model/__init__.py



DONE In batch processing: create checkboxes for the structure categories, so users can seelct to, e.g., not analyze sarcomere domains.
     No checkboxes needed for motion.

data export: separate buttons for structure and motion, each in the structure analysis and motion analysis sections at the top below the upper button,
and remove the metadata part.(in data export not the one in main window)

DONE the loi detection now also has multi-segment lines ( see plot_lois in Plots.py). The Gui should also show them as multi-segment lines
     HAS TO BE TESTED

DONE rename the button analyze structure at the top to full analysis structure

motion analysis: 
DONE predict and analyze contraction cycles: add default path analogous to the other model weights(model_ContractionNet.pt in models dir)
DONE calculate and analyze...: the default params of smoothing filter V(t) are wrong, see first point above.
DONE add a button at the bottom for plotting a summary using plot_loi_summary_motion in Plots.py that then pops up in a new window (use the matplotlib backend for qt)




"""