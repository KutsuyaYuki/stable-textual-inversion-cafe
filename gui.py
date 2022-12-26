import sys
import json
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QLabel, QGridLayout
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from pytorch_lightning import Trainer
import main
import os

# Create the main window
app = QApplication(sys.argv)

# Force the style to be the same on all OSs:
app.setStyle("Fusion")

# Now use a palette to switch to dark colors:
palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, Qt.white)
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, Qt.black)
palette.setColor(QPalette.ToolTipText, Qt.white)
palette.setColor(QPalette.Text, Qt.white)
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, Qt.white)
palette.setColor(QPalette.BrightText, Qt.red)
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, Qt.black)
app.setPalette(palette)


# Set the style of the application to a dark style
app.setStyle(QStyleFactory.create("QtDarkBlueStyle"))

window = QWidget()
window.setWindowTitle("GUI for Cafe's repo")
window.resize(700, 200)

# Create two input fields and a button
input_yaml          = QLineEdit()
input_ckpt          = QLineEdit()
input_dataset       = QLineEdit()
input_init          = QLineEdit()
input_projectname   = QLineEdit()
button_yaml         = QPushButton("...")
button_ckpt         = QPushButton("...")
button_dataset      = QPushButton("...")
button_submit       = QPushButton("Submit")

# Create labels for the input fields
label_yaml          = QLabel("Config file (YAML):")
label_ckpt          = QLabel("Model (CKPT):")
label_dataset       = QLabel("Input (Dataset):")
label_init          = QLabel("Initword:")
label_projectname   = QLabel("Project name:")

# Set the input fields and file dialog to the stored values, if they exist
# Check if the JSON file exists
if os.path.exists("gui_config.json"):
    # Read the stored values from the JSON file
    with open("gui_config.json", "r") as f:
        stored_values = json.load(f)
    # Set the input fields and file dialog to the stored values
        input_yaml.setText(stored_values.get("input_yaml", ""))
        input_ckpt.setText(stored_values.get("input_ckpt", ""))
        input_dataset.setText(stored_values.get("input_dataset", ""))
        input_init.setText(stored_values.get("input_init", ""))
        input_projectname.setText(stored_values.get("input_projectname", ""))
else:
    # Create the JSON file with the default values
    with open("gui_config.json", "w") as f:
        json.dump({
        "input_yaml": "", 
        "input_ckpt": "", 
        "input_dataset": "", 
        "input_init": "", 
        "input_projectname": ""}, f)

# Create a grid layout to hold the widgets
layout = QGridLayout()

# Add the input fields and labels to the grid
layout.addWidget(label_yaml, 0, 0)
layout.addWidget(input_yaml, 0, 1)
layout.addWidget(button_yaml, 0, 2)

button_yaml.clicked.connect(lambda : input_yaml.setText(QFileDialog.getOpenFileName(caption="Select Configuration")[0]))

layout.addWidget(label_ckpt, 1, 0)
layout.addWidget(input_ckpt, 1, 1)
layout.addWidget(button_ckpt, 1, 2)

button_ckpt.clicked.connect(lambda : input_ckpt.setText(QFileDialog.getOpenFileName(caption="Select Model")[0]))

layout.addWidget(label_dataset, 2, 0)
layout.addWidget(input_dataset, 2, 1)
layout.addWidget(button_dataset, 2, 2)

button_dataset.clicked.connect(lambda : input_dataset.setText(QFileDialog.getExistingDirectory(caption="Select Dataset Folder")))

layout.addWidget(label_init, 3, 0)
layout.addWidget(input_init, 3, 1)

layout.addWidget(label_projectname, 4, 0)
layout.addWidget(input_projectname, 4, 1)

# Add the button to the grid
layout.addWidget(button_submit, 5, 0, 1, 2)

# Set the layout of the main window
window.setLayout(layout)

def save_data_to_json():
    values = {
        "input_yaml": input_yaml.text(), 
        "input_ckpt": input_ckpt.text(), 
        "input_dataset": input_dataset.text(), 
        "input_init": input_init.text(), 
        "input_projectname": input_projectname.text()}

    # Write the dictionary to a JSON file
    with open("gui_config.json", "w") as f:
        json.dump(values, f)

# Define a function to be called when the button is clicked
def on_button_submit_clicked():
    save_data_to_json()
    parser = main.get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args([
        '--base',input_yaml.text(),
        '--train','true',
        "--no-test",'true',
        "--actual_resume",input_ckpt.text(),
        "--gpus","0,",
        "--data_root",input_dataset.text(),
        "--init_word",input_init.text(),
        "-n",input_projectname.text()
        ])
    main.main(opt, unknown)

# Connect the clicked signal of the button to the function we just defined
button_submit.clicked.connect(on_button_submit_clicked)


def on_close(self):
    # save the data to a JSON file
    save_data_to_json()

window.closeEvent = on_close


# Show the main window
window.show()

# Run the PyQt event loop
sys.exit(app.exec_())
