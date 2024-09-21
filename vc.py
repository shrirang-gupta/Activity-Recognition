import numpy as np
import cv2
import os 
import pandas as pd

train_label_path=r"E:\content\Hollywood2_final\labels\train"
train_video_path=r"E:\content\Hollywood2_final\videos\train"


N_FRAMES = 10 
def get_video_array(video_path):
    #load video from video_path from cv2
    video = cv2.VideoCapture(video_path)
    frames = []
    
    while (video.isOpened()):
        ret, frame = video.read() 
        if ret:
            # if video.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0:
            #     re_frame = cv2.resize(frame, (224, 224))
            #     frames.append(re_frame)
            if len(frames) < N_FRAMES:
                frame = cv2.resize(frame, (32, 32))
                frame = frame.astype(np.float32) 
                frame = frame / 255.0
                frames.append(frame)
        else:
            break
    frames = np.array(frames)

    return frames


if __name__ == "__main__":
    columns = ['videoname','AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake', 'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']
    #create ampty dataframe with columns
    # array_path=r"E:\content\Hollywood2_final\labels\array.npy"
    # arr_path = "array.npz"
    df = pd.DataFrame(columns=columns)
    print(df.shape)

    #create empty nd array
    frames = np.zeros((len(os.listdir(train_video_path)),N_FRAMES,32,32,3))
    labels = []
    #loop over all txt files in label_path
    j=0
    for index,video in enumerate(os.listdir(train_video_path)):
        video_path=os.path.join(train_video_path, video)
        frame = get_video_array(video_path)
        #add frame to frames array

        frames[index]=frame

    with open('video_array_small.npy', 'wb') as f:
        np.save(f, frames)


    for files_label in os.listdir(train_label_path):
        #open the txt file and read the lines
        print("files_label",files_label)
        #split label name to get video name
        label = files_label.split(".")[0]
        with open(os.path.join(train_label_path, files_label), "r") as f:
            lines = f.readlines()
        #loop over all lines in the txt file
        i=0
        for line in lines:
            vid,l1=line.split()
            df.loc[i,'videoname']=vid
            if l1=='1':
                df.loc[i,label]=1
            else:
                df.loc[i,label]=0
            i=i+1

        
    print(df.shape)
    df.to_csv(r"E:\content\Hollywood2_final\labels\train3.csv",index=False)



            











