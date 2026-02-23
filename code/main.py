"""

  Application: Data processing software for RHESSI and STIX instruments

  Start date: 01/03/2021
    
  Last update: 14/04/2025 (Starting date) by Assamoua Koman and Abdallah Hamini
  Version: 0.1

  Creators: Théo Pillevesse, Diane Chan Sock Line, Liaisian Abdrakhmanova, Abdallah Hamini, Aichatou Aboubacar

  Organization: LESIA, Observatory of Paris, France

  Graphical User Interface: GUI was created using tkinter library

  Usage: information to test the program provided in Requirements file

  Status = 'Development'

"""

# Importations
import webbrowser
from tkinter import *
from tkinter import messagebox
import second
import background
import do_fit
import fit_all


# ========================= Functions to activate the menu bars and menu options =========================


def SelectInput():
    """Creates a new window called 'Select Input'. Plots data for rate, counts and flux in different profiles."""
    second.SecondWindow(root)


def SelectBackground():
    """Creating a new window with widgets and options to select background, and display new plots."""
    background.BackgroundWindow(root)


def Fitting():
    """Creating a new window where user can select function to fit and plot (counts data)."""
    do_fit.Fitting(root)

def ResponseFitting():
    """Creating a new window where user can select function to fit and plot with matrix response."""
    fit_all.Fitting(root)

def STIX_Guide():
    """Opens up HTML version of the OSPEX documentation using default browser."""
    url2 = "https://lesia.obspm.fr/STIX-description-de-l-instrument.html"
    webbrowser.open(url2)


def clickedContact():
    """Displays the information about creators and contacts in "Help" menu option."""
    messagebox.showinfo('STIX Contact Information', 'The STIX package was developed by  Abdallah Hamini'
                                                    ' at LESIA, Paris Observatory, France'
                                                    '\n \n@:abdallah.hamini@obspm.fr'
                                                    '\n\n№:(+33)145077470')


def clickedHelp_on_Help():
    """Adding the description for Help on Help menu option."""
    messagebox.showinfo('STIX Help Information', 'The documentation for STIX is in HTML format.'
                                                 '\n\nWhen you click the help buttons, your preferred browser will be '
                                                 'activated.'
                                                 '\n\nThe browser may start in iconized mode.'
                                                 '\nIf it does not appear, you may need to find it on the taskbar.')


# ========================= Creating main window =========================

root = Tk()
root.title('STIX Main Window')
# root.iconbitmap(r"/home/stage/PycharmProjects/testing/Rhessi.ico")
root.geometry("500x600")
mainmenu = Menu(root)
root.config(menu=mainmenu)

Label(root,
      text="\n \n \nSTIX",
      fg="red",
      font="Helvetica 12 bold italic").pack()

Label(root,
      text="\n \n \n \n Spectral Data Analysis Package",
      fg="red",
      font="Times").pack()

Label(root,
      text="\n \n \n Use the buttons under File to: "
           "\n \n 1. Select Input Data Files"
           "\n 2. Define Background and Analysis Intervals, \n and Select Fit Function Components"
           "\n 3. Fit data "
           "\n 4. View Fit Results "
           "\n 5. Save Session and Results",
      fg="blue",
      font="Times",
      justify='left').pack()


# ============================ Adding the menu bars in main window ============================

filemenu = Menu(mainmenu, tearoff=0)
windowmenu = Menu(mainmenu, tearoff=0)
helpmenu = Menu(mainmenu, tearoff=0)

filemenu.add_command(label="Select Input ...", command=SelectInput)
filemenu.add_command(label="Select Background ...", command=SelectBackground)
# filemenu.add_command(label="Select Fit Options and Do Fit (Test) ...", command=Fitting)
filemenu.add_command(label="Plot Fit Results ...", command=ResponseFitting)
filemenu.add_command(label="Set parameters manually ...")
filemenu.add_command(label="Set parameters from script")

filemenu.add_separator()

filemenu.add_command(label="Setup Summary")
filemenu.add_command(label="Fit Results Summary")
filemenu.add_command(label="Write script")
filemenu.add_command(label="Save Fit Results (No Script)")
filemenu.add_command(label="Import Fit Results")
filemenu.add_command(label="Write FITS Spectrum File")

filemenu.add_separator()

filemenu.add_command(label="Clear Stored Fit Results")
filemenu.add_command(label="Reset Entire OSPEX Session to Defaults")

filemenu.add_separator()

filemenu.add_command(label="Set Plot Preferences")

filemenu.add_separator()

filemenu.add_command(label="Configure Plot File")
filemenu.add_command(label="Create Plot File")

filemenu.add_separator()

filemenu.add_command(label="Select Printer ...")

filemenu.add_separator()

filemenu.add_command(label="Configure Print Plot Output...")
filemenu.add_command(label="Print Plot")

filemenu.add_separator()

filemenu.add_command(label="Export Data")
filemenu.add_command(label="Reset Widgets (Recover from Problems)")
filemenu.add_command(label="Exit", command=root.quit)

windowmenu.add_command(label="Current Panel")
windowmenu.add_command(label="Show All Panels")
windowmenu.add_command(label="2x2 Panels")
windowmenu.add_command(label="Delete All Panels")
windowmenu.add_command(label="Multi-Panel Options")

helpmenu.add_command(label="STIX Guide", command=STIX_Guide)
helpmenu.add_command(label="Contacts", command=clickedContact)
helpmenu.add_command(label="Help on Help", command=clickedHelp_on_Help)

mainmenu.add_cascade(label="File", menu=filemenu)
mainmenu.add_cascade(label="Window_Control", menu=windowmenu)
mainmenu.add_cascade(label="Help", menu=helpmenu)

root.mainloop()
