import customtkinter as ctk
import webbrowser
import second
import background
import do_fit

# Initialisation de CustomTkinter
ctk.set_appearance_mode("dark")  # Mode sombre
#ctk.set_default_color_theme("blue")  # Thème de couleur

# ========================= Fonctions liées aux menus =========================

def SelectInput():
    """Crée une nouvelle fenêtre pour afficher des profils (rate, counts, flux)."""
    second.SecondWindow(root)

def SelectBackground():
    """Crée une fenêtre pour sélectionner le fond et afficher de nouveaux graphes."""
    background.BackgroundWindow(root)

def SelectFitting():
    """Crée une fenêtre pour effectuer un ajustement (fit) sur les données."""
    do_fit.DoFitWindow(root)

def open_website():
    webbrowser.open("https://www.lesia.obspm.fr")

# ========================= Fenêtre principale =========================

root = ctk.CTk()
root.title("Data Analysis Tool - RHESSI / STIX")
root.geometry("800x600")

# ========================= Barre de menu =========================

menu_bar = ctk.CTkFrame(root)
menu_bar.pack(pady=20)

btn_input = ctk.CTkButton(menu_bar, text="Input", command=SelectInput)
btn_input.grid(row=0, column=0, padx=10)

btn_background = ctk.CTkButton(menu_bar, text="Background", command=SelectBackground)
btn_background.grid(row=0, column=1, padx=10)

btn_fit = ctk.CTkButton(menu_bar, text="Fitting", command=SelectFitting)
btn_fit.grid(row=0, column=2, padx=10)

btn_about = ctk.CTkButton(menu_bar, text="About LESIA", command=open_website)
btn_about.grid(row=0, column=3, padx=10)

# ========================= Texte ou zone principale =========================

main_label = ctk.CTkLabel(root, text="Bienvenue dans l'outil d'analyse des données RHESSI / STIX",
                           font=("Helvetica", 18), text_color="white")
main_label.pack(pady=40)

# ========================= Lancement =========================

root.mainloop()
