from tkinter import * 
import tkinter.font as font
from PIL import ImageTk,Image
from tkinter import messagebox
from functools import partial
import sys
import os


top = Tk()
top.title("FASHION APPAREL LOGIN")
FILENAME='img.jpg'
input_name=StringVar()
input_pswd=StringVar()
#set width and height
myFont = font.Font(family='Courier', size=15, weight='bold')
canvas=Canvas(top,width=1920,height=1080)
canvas.pack()
tk_img = ImageTk.PhotoImage(file = FILENAME)
canvas.create_image(800,510,image=tk_img)

top.geometry("1920x1080")   

def validatelogin(username,password):
    
    login={"MITHUN":"30","SANJIV":"48","VIJAY":"58","SHREEVATSA":"49"}
    if username.get() in login:
        if password.get() in login[username.get()]:
            os.system('GUI.py')
        else:
            messagebox.showwarning("showwarning", "Invalid Input")
    else:
        messagebox.showwarning("showwarning", "Invalid Input")
    
        

fa = Label(top,text = "FASHION APPAREL DETECTION SYSTEM",font = "GAZZARELLI 30 bold").place(x = 200, 
                                           y = 200)       
    
# the label for user_name  
user_name = Label(top,  
                  text = "Username",font = "Courier 20 bold").place(x = 500, 
                                           y = 300)   
    
# the label for user_password   
user_password = Label(top,  
                      text = "Password",font = "Courier 20 bold").place(x = 500, 
                                               y = 350)   
    
user_name_input_area = Entry(top, 
                             width = 30,textvariable = input_name, justify = CENTER).place(x = 650, 
                                               y = 310)   
    
user_password_entry_area = Entry(top, 
                                 width = 30,textvariable = input_pswd,show='*', justify = CENTER).place(x = 650, 
                                                   y = 355)

validatelogin = partial(validatelogin,input_name, input_pswd)

submit_button = Button(top, text = "Submit",command=validatelogin,anchor='c',width=10)
submit_button.place(relx = 0.5, x =-120, y = 420, anchor =CENTER )

quit_button = Button(top, text = "Quit", command = top.destroy, anchor = 'c',width=10)
quit_button.place(relx = 0.5, x =50, y =420, anchor = CENTER)

quit_button['font']=myFont
submit_button['font']=myFont

top.mainloop()
