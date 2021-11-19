# Mapping and perception for autonomous robot , semester A 2021/22
# Roy orfaig & Ben-Zion Bobrovsky
# Project 2
# KF, EKF and EKF-SLAM! 

import os
from data_loader import DataLoader
from project_questions import ProjectQuestions


if __name__ == "__main__":
    # date = '2011_09_30' #example (fill your correct data)
    # drive = '0033' #The recording number I used in the sample in class  (fill your correct data)
    # basedir = "../../Project 1/Data/kitti_data"
    # date = '2011_09_26'
    # drive = '0013'
    basedir = "../Data/kitti_data"
    date = '2011_09_30'
    drive = '0027'
    data_dir = os.path.join(basedir, "Ex3_data")
    
    dataset = DataLoader(basedir, date, drive, data_dir)
    
    project = ProjectQuestions(dataset)
    project.run()