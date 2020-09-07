import os
import time
import threading
from tkinter import *
from tkinter import Button, Tk, HORIZONTAL
from tkinter.ttk import Progressbar
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
import search_bing_api

def percentageCalculator(x, y, case=2):
    """Calculate percentages
       Case1: What is x% of y?
       Case2: x is what percent of y?
       Case3: What is the percentage increase/decrease from x to y?
    """
    if case == 1:
        #Case1: What is x% of y?
        r = x/100*y
        return r
    elif case == 2:
        #Case2: x is what percent of y?
        r = (x+1)/y*100
        return r
    elif case == 3:
        #Case3: What is the percentage increase/decrease from x to y?
        r = (y-x)/x*100
        return r
    else:
        raise Exception("Only case 1,2 and 3 are available!")

def makeform(root, fields):
    entries = {}
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=30, text=field+": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, "0")
        row.pack(side=tk.TOP, 
                 fill=tk.X, 
                 padx=5, 
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, 
                 expand=tk.YES, 
                 fill=tk.X)
        entries[field] = ent
    return entries

infoDict = []
##def processEntry(entries):
##        
##    return infoDict 
    
    
def runActions(progress, status,entries):
    e1=(entries['Enter number of photos:'].get())
    infoDict.append(int(e1))
    e2=(entries['Enter the apperal type(eg.\"black shirt\"):'].get())
    infoDict.append(e2)
    e3=filedialog.askdirectory(initialdir = "C:\\Users\\Mithun\\Desktop\\8th sem\\Major Project\\FASSION APPEARAL\\keras-multi-label\\dataset",title = "Select file")
    alist = infoDict[0]
    log = open("log.txt", "a")
    os.system('search_bing_api.py e1 e2 e3')
    try:

        p = 1
        for i in range(1,alist+1):
            # Case2: x is what percent of y?
            unit = percentageCalculator(p, alist+1, case=2)
            p += 1

            #TODO make a decorator!
            time.sleep(10) #some func


            step = "Working on {}".format(i) 
            log.write(str('\n[OK]'))
            progress['value'] = unit
            percent['text'] = "{}%".format(int(unit))
            status['text'] = "{}".format(step)

            root.update()

        messagebox.showinfo('Info', "Process completed!")
        sys.exit()


    except Exception as e:
        messagebox.showinfo('Info', "ERROR: {}".format(e))
        sys.exit()

    log.close()






root = Tk()
root.title("Download Images..")
root.geometry("600x320")

fields = ('Enter number of photos:', 'Enter the apperal type(eg.\"black shirt\"):')

ents = makeform(root, fields)
runButton = Button(root, text='Start downloading', command=(lambda e=ents: runActions(progress, status,e)))
percent = Label(root, text="", anchor=S) 
progress = Progressbar(root, length=500, mode='determinate')    
status = Label(root, text="Click button to start process..", relief=SUNKEN, anchor=W, bd=2) 

runButton.pack(pady=15)
percent.pack()
progress.pack()
status.pack(side=BOTTOM, fill=X)

root.mainloop()
