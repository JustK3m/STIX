from tkinter import *


class EntryInt(Entry):
    """This class uses the 'Entry' funtion included in the tkinter library. It allows to write only integers as entries,
    with the exception of the blank '' Entry box. It works the same way as Entry function from tkinter."""
    def __init__(self, master=None, **kwargs):
        self.var = StringVar()
        Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check_int)
        self.get, self.set = self.var.get, self.var.set

    def check_int(self, *args):
        """This is the function that can be used to replace Entry() from tkinter."""
        if self.get().isdigit() or self.get() == '':
            # Current value is only digits, or is empty; allow this
            self.old_value = self.get()
        else:
            # There's non-digit characters in the input; reject this
            self.set(self.old_value)


if __name__ == '__main__':

    print("You should not be able to enter other caracters than numeric caracters.")
    print("Try entering non-digits such as letters, then digits.")

    window = Tk()
    From_entry = EntryInt(window, width=25)
    From_entry.grid(column=1, row=2, padx=5)
    window.mainloop()
