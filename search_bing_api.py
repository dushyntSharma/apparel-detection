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

    API_KEY = "381dc65ef1bf4fdba651a9c51dc0a456"
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
