import tkinter as tk

x, y = 535, 395
w, h = 530, 525

root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)
root.geometry(f"{w}x{h}+{x}+{y}")
root.attributes("-alpha", 0.3)

frame = tk.Frame(root, bg="red", highlightthickness=2, highlightbackground="red")
frame.pack(fill=tk.BOTH, expand=True)

root.bind("<Escape>", lambda e: root.destroy())  # Press ESC to close
root.mainloop()

