import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import sys
from smartShop import smartShop
from database import collection

stop_event = None
thread = None

def run_tracking():
    global stop_event
    smartShop(stop_event)

def start_live_feed():
    global thread, stop_event
    if thread and thread.is_alive():
        messagebox.showinfo("Info", "Live feed is already running.")
        return

    stop_event = threading.Event()
    thread = threading.Thread(target=run_tracking)
    thread.start()

def stop_live_feed(silent=False):
    global stop_event, thread
    if thread and thread.is_alive():
        stop_event.set()
        thread.join()
        cv2.destroyAllWindows()
        if not silent:
            messagebox.showinfo("Info", "Live feed stopped.")
    else:
        if not silent:
            messagebox.showinfo("Info", "Live feed is not running.")

def show_customers():
    customers = list(collection.find({}))
    if not customers:
        messagebox.showinfo("Info", "No customers found.")
        return

    win = tk.Toplevel()
    win.title("Stored Customers")

    cols = ["Name", "Age Range", "Gender", "Visits", "Last Seen"]
    tree = ttk.Treeview(win, columns=cols, show='headings')
    for col in cols:
        tree.heading(col, text=col)

    for cust in customers:
        name = cust.get("name", "Unknown")
        age_range = f"{cust.get('min_age', '?')} - {cust.get('max_age', '?')}"
        gender = cust.get("gender", "Unknown") if isinstance(cust.get("gender"), str) else "Unknown"
        visits = cust.get("visit_count", 0)
        last_seen = cust.get("last_seen", "N/A")
        tree.insert('', 'end', values=(name, age_range, gender, visits, last_seen))

    tree.pack(expand=True, fill='both')

    # Add scrollbar
    scrollbar = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')

def quit_app():
    stop_live_feed(silent=True)
    sys.exit()

def on_close():
    stop_live_feed(silent=True)
    root.destroy()

def build_gui():
    global root
    root = tk.Tk()
    root.title("SmartShop Tracker")
    root.geometry("400x250")

    title = tk.Label(root, text="SmartShop Tracker", font=("Helvetica", 16, "bold"))
    title.pack(pady=10)

    start_btn = tk.Button(root, text="Start Live Feed", command=start_live_feed, width=20, bg="green", fg="white")
    start_btn.pack(pady=5)

    stop_btn = tk.Button(root, text="Stop Live Feed", command=stop_live_feed, width=20, bg="red", fg="white")
    stop_btn.pack(pady=5)

    show_btn = tk.Button(root, text="Show Stored Customers", command=show_customers, width=25)
    show_btn.pack(pady=10)

    exit_btn = tk.Button(root, text="Exit", command=quit_app, width=20)
    exit_btn.pack(pady=5)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    build_gui()
