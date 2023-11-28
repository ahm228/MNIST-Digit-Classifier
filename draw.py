import tkinter as tk
from PIL import Image, ImageDraw

class DrawingApp:
    def __init__(self, root):
        self.root = root
        root.title("Drawing Pad with Data Display")

        self.canvas_width = 400
        self.canvas_height = 300

        # Create a drawing canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Create an image to draw on
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to canvas
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create a label to display data
        self.data_label = tk.Label(root, text="Your Data Here", font=("Arial", 14))
        self.data_label.pack()

        # Button to clear canvas
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.old_x = None
        self.old_y = None

    def paint(self, event):
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line((self.old_x, self.old_y, x, y), width=2, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, x, y), fill='black', width=2)

        self.old_x = x
        self.old_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
