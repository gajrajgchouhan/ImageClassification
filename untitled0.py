
import os
import glob

os.chdir(r"/home/gajraj/Downloads/MLChallenge/horses")
for index, oldfile in enumerate(glob.glob("*.png"), start=1):
    newfile = 'human-{:3}.jpg'.format(index)
    os.rename (oldfile,newfile)