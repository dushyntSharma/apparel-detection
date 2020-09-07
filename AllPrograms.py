#LOGIN.py


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




#captureimage.py



# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle

# import the necessary packages
import warnings
print("Running Software...")
warnings.filterwarnings("ignore")
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


### construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-m", "--model", required=True,
##	help="path to trained model model")
##ap.add_argument("-l", "--labelbin", required=True,
##	help="path to label binarizer")
####ap.add_argument("-i", "--image", required=True,
####	help="path to input image")
##args = vars(ap.parse_args())
##print(args)
cap=cv2.VideoCapture(0)
print(cap)

while True :
        # load the image
        ret,image = cap.read()
        cv2.imshow("Capturing", image)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            output = imutils.resize(image, width=400)
            # pre-process the image for classification
            image = cv2.resize(image, (96, 96))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
    
            # load the trained convolutional neural network and the multi-label
            # binarizer
            warnings.filterwarnings("ignore")
            print("[INFO] loading network...")
            model = load_model('fashion.model')
           
            mlb = pickle.loads(open('mlb.pickle', "rb").read())
    
            # classify the input image then find the indexes of the two class
            # labels with the *largest* probability
            warnings.filterwarnings("ignore")
            print("[INFO] classifying image...")
            proba = model.predict(image)[0]
            
            idxs = np.argsort(proba)[::-1][:2]
    
            # loop over the indexes of the high confidence class labels
            warnings.filterwarnings("ignore")
            for (i, j) in enumerate(idxs):
                    # build the label and draw the label on the image
                    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
                    print(label)

                    cv2.putText(output, label, (10, (i * 30) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # show the probabilities for each of the individual labels

 
            for (label, p) in zip(mlb.classes_, proba):
	            print("{}: {:.2f}%".format(label, p * 100))
             
    
            # show the output image
            cv2.imshow("Output", output)            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        elif key == ord('q'):
            print("Turning off camera.")
            cap.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break




#GUI.py




import sys
import os
from tkinter import *
from PIL import ImageTk,Image
import tkinter.font as font
from tkinter import messagebox
import random

window=Tk(className="Fashion Apparel")
frame = Frame(window) 
frame.pack() 

FILENAME='img.jpg'

#set width and height
myFont = font.Font(family='Courier', size=20, weight='bold')
canvas=Canvas(window,width=1920,height=1080)
canvas.pack()
tk_img = ImageTk.PhotoImage(file = FILENAME)
canvas.create_image(800,510,image=tk_img)


fa = Label(window,text = "FASHION APPAREL DETECTION SYSTEM",font = "GAZZARELLI 30 bold").place(x = 200, 
                                           y = 200)    

quit_button = Button(window, text = "Quit", command = window.destroy, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
quit_button.place(relx = 0.5, x =200, y =550, anchor = CENTER)

window.geometry('1920x1080')

def runclassify1():
    messagebox.showinfo("Message", "Please wait, we are loading the software.")
    os.system('CaptureImage.py')
def runclassify():
    messagebox.showinfo("Message", "Please wait, we are loading the software.")
    os.system('LiveVideo.py')
def runimport():
    messagebox.showinfo("Message", "Please wait, we are loading the software.")
    os.system('importimage.py')
def cdataset():
    messagebox.showinfo("Message", "Please wait, we are loading the software.")
    os.system('search_bing_api.py')
    sys.exit()
def trainm():
    con=messagebox.askokcancel("Confirm", "Training the Model will take time(approx. 90min).")
    if(con==True):
        os.system('train.py')
    else:
        return

capimg = Button(window, text = "Click Picture", command = runclassify1, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
capimg.place(relx = 0.5, x =200, y = 350, anchor = CENTER)


liveimg = Button(window, text = "Live Video", command = runclassify, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
liveimg.place(relx = 0.5, x =-220, y = 350, anchor =CENTER )

impimg = Button(window, text = "Select Image", command = runimport, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
impimg.place(relx = 0.5, x =-220, y = 450, anchor =CENTER )

dataset = Button(window, text = "Create Dataset", command = cdataset, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
dataset.place(relx = 0.5, x =200, y = 450, anchor =CENTER )

train = Button(window, text = "Train Model", command = trainm, anchor = 'c',
                    width = 15, activebackground = "#33B5E5")
train.place(relx = 0.5, x =-220, y = 550, anchor =CENTER )


train['font']=myFont
quit_button['font'] = myFont
capimg['font'] = myFont
liveimg['font'] = myFont
impimg['font'] = myFont
dataset['font']=myFont

window.mainloop()







#importimage.py






# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os 
from tkinter import filedialog

dirname = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

# construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-m", "--model", required=True,
##	help="path to trained model model")
##ap.add_argument("-l", "--labelbin", required=True,
##	help="path to label binarizer")
##ap.add_argument("-i", "--image", required=True,
##	help="path to input image")
##args = vars(ap.parse_args())

# load the image
image = cv2.imread(dirname)
output = imutils.resize(image, width=400)
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model('fashion.model')
mlb = pickle.loads(open('mlb.pickle', "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))


# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)








#LiveVideo.py





# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle

# import the necessary packages
import warnings
print("Running Software...")
warnings.filterwarnings("ignore")
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-m", "--model", required=True,
##	help="--fashion.model")
##ap.add_argument("-l", "--labelbin", required=True,
##	help="--mlb.pickle")
##ap.add_argument("-i", "--image", required=True,
##	help="path to input image")
##args = vars(ap.parse_args())
cap=cv2.VideoCapture(0)
print(cap)

while True :
        # load the image
        ret,image = cap.read()
        output = imutils.resize(image, width=400)
         
        # pre-process the image for classification
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network and the multi-label
        # binarizer
        print("[INFO] loading network...")
        model = load_model('fashion.model')
        mlb = pickle.loads(open('mlb.pickle', "rb").read())

        # classify the input image then find the indexes of the two class
        # labels with the *largest* probability
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:2]

        # loop over the indexes of the high confidence class labels
        for (i, j) in enumerate(idxs):
                # build the label and draw the label on the image
                label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
                cv2.putText(output, label, (10, (i * 30) + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # show the probabilities for each of the individual labels
        for (label, p) in zip(mlb.classes_, proba):
                print("{}: {:.2f}%".format(label, p * 100))

        # show the output image
        cv2.imshow("Output", output)
        if cv2.waitKey(100) & 0xFF == ord('q'):
                break










#search_bin_api.py











# USAGE
# python search_bing_api.py --query "blue jeans" --output dataset/blue_jeans
# python search_bing_api.py --query "blue dress" --output dataset/blue_dress
# python search_bing_api.py --query "red dress" --output dataset/red_dress
# python search_bing_api.py --query "red shirt" --output dataset/red_shirt
# python search_bing_api.py --query "blue shirt" --output dataset/blue_shirt
# python search_bing_api.py --query "black jeans" --output dataset/black_jeans

# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import Button, Tk, HORIZONTAL
from tkinter.ttk import Progressbar
from tkinter import simpledialog
from tkinter import ttk as ttk
from tkinter import messagebox
import time


# construct the argument parser and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-q", "--query", required=True,
##  help="search query to search Bing Image API for")
##ap.add_argument("-o", "--output", required=True,
##  help="path to output directory of images")
##args = vars(ap.parse_args())

# set your Microsoft Cognitive Services API key along with (1) the
# maximum number of results for a given search and (2) the group size
# for results (maximum of 50 per request)
url=["Using Cognitive Search..."]
def searchbing(e1,e2,e3):
        
##    ROOT = tk.Tk()
##
##    ROOT.withdraw()
    # the input dialog
    #e1 = 10#int(simpledialog.askstring(title="Create Dataset",
                                      #prompt="Enter number of photos: "))
    #e2 = "green shirt"#simpledialog.askstring(title="Create Dataset",
                                      #prompt="Enter the apperal type(eg.\"black shirt\"): ")

    API_KEY = "a028992ad9394714a76202676876d290"
    MAX_RESULTS = int(e1)
    GROUP_SIZE = MAX_RESULTS

    # set the endpoint API URL
    URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search/"

    # when attemping to download images from the web both the Python
    # programming language and the requests library have a number of
    # exceptions that can be thrown so let's build a list of them now
    # so we can filter on them
    EXCEPTIONS = set([IOError, FileNotFoundError,
        exceptions.RequestException, exceptions.HTTPError,
        exceptions.ConnectionError, exceptions.Timeout])

    # store the search term in a convenience variable then set the
    # headers and search parameters
    term = e2       #input('Enter the apperal type:') #"black shirt" #args["query"]
    headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
    params = {"q": term, "offset": 0, "count": GROUP_SIZE}

##    root = tk.Tk()
    dirname =e3
    #"C:\\Users\\Mithun\\Desktop\\8th sem\\Major Project\\FASSION APPEARAL\\keras-multi-label\\dataset\\green_shirt"
    # filedialog.askdirectory(initialdir = "C:\\Users\\Mithun\\Desktop\\8th sem\\Major Project\\FASSION APPEARAL\\keras-multi-label\\dataset",title = "Select file")

    # make the search
    print("[INFO] searching Bing API for '{}'".format(term))
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()

    # grab the results from the search, including the total number of
    # estimated results returned by the Bing API
    results = search.json()
    estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
    print("[INFO] {} total results for '{}'".format(estNumResults,
        term))


    # initialize the total number of images downloaded thus far
    total = 0

    # loop over the estimated number of results in `GROUP_SIZE` groups
    for offset in range(0, estNumResults, GROUP_SIZE):
        # update the search parameters using the current offset, then
        # make the request to fetch the results
        print("[INFO] making request for group {}-{} of {}...".format(
            offset, offset + GROUP_SIZE, estNumResults))
        params["offset"] = offset
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print("[INFO] saving images for group {}-{} of {}...".format(
            offset, offset + GROUP_SIZE, estNumResults))

        # loop over the results
        for v in results["value"]:
                # try to download the image
                try:
                        # make a request to download the image
                        print("[INFO] fetching: {}".format(v["contentUrl"]))
                        url.append(format(v["contentUrl"]))
                        r = requests.get(v["contentUrl"], timeout=30)

                        # build the path to the output image
                        ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                        p = os.path.sep.join([dirname, "{}{}".format(
                                str(total).zfill(8), ext)]) #args[output]

                        # write the image to disk
                        f = open(p, "wb")
                        f.write(r.content)
                        f.close()

                # catch any errors that would not unable us to download the
                # image
                except Exception as e:
                        # check to see if our exception is in our list of
                        # exceptions to check for
                        if type(e) in EXCEPTIONS:
                                print("[INFO] skipping: {}".format(v["contentUrl"]))
                                continue

                # try to load the image from disk
                image = cv2.imread(p)

                # if the image is `None` then we could not properly load the
                # image from disk (so it should be ignored)
                if image is None:
                        print("[INFO] deleting: {}".format(p))
                        os.remove(p)
                        continue

                # update the counter
                total += 1
        url.append("Download Complete")

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
def check(n):
    try:
        val = int(n)
        return False
    except ValueError:
        try:
            val = float(n)
            return False
        except ValueError:
            return True
        
def runActions(progress, status,entries):
    e1=(entries['Enter number of photos:'].get())
    infoDict.append(int(e1))
    e2=(entries['Enter the apperal type(eg.\"black shirt\"):'].get())
    infoDict.append(e2)
        #os.system('search_bing_api.py e1 e2 e3')
    if(e1!='0' and check(e2)):
        e3=filedialog.askdirectory(initialdir = "C:\\Users\\Mithun\\Desktop\\8th sem\\Major Project\\FASSION APPEARAL\\keras-multi-label\\dataset",title = "Select file")
        alist = infoDict[0]
        log = open("log.txt", "a")

        try:

            p = 1
            searchbing(e1,e2,e3)
            for i in range(1,alist+1):
                # Case2: x is what percent of y?
                unit = percentageCalculator(p, alist+1, case=2)
                p += 1
                time.sleep(2) #some func

                step = url[i-1]   #"Working on {}".format(i)
                log.write(str('\n[OK]'))
                progress['value'] = unit
                percent['text'] = "{}%".format(int(unit))
                status['text'] = "{}".format(step)
                 

                root.update()
            messagebox.showinfo('Info', "Process completed!")
            unit=0
            progress['value'] = unit
            percent['text'] = "{}%".format(int(unit))
            status['text'] = "{}".format("Download Complete")
            sys.exit()


        except Exception as e:
            messagebox.showinfo('Info', "ERROR: {}".format(e))
            sys.exit()
    else:
        messagebox.showinfo('Info', "Enter the valid Input's.")
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
status = Label(root, text="Click the start button to begin downloads...", relief=SUNKEN, anchor=W, bd=2) 

runButton.pack(pady=15)
percent.pack()
progress.pack()
status.pack(side=BOTTOM, fill=X)

root.mainloop()












#train.py






# USAGE
# python train.py -d dataset -m fashion.model -l mlb.pickle -p plot

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import warnings
print("Training the images...")
warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
#import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-d", "--dataset", required=True,
##	help="path to input dataset (i.e., directory of images)")
##ap.add_argument("-m", "--fashion.model", required=True,
##	help="path to output model")
##ap.add_argument("-l", "--mlb.pickle", required=True,
##	help="path to output label binarizer")
##ap.add_argument("-p", "--plot", type=str, default="plot.png",
##	help="path to output accuracy/loss plot")
##args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 10
INIT_LR = 1e-2
BS = 30
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images('C:\\Users\\Mithun\\Desktop\\8th sem\\Major Project\\FASSION APPEARAL\\keras-multi-label\\dataset')))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('fashion.model')

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('mlb.pickle', "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('plot.png')



