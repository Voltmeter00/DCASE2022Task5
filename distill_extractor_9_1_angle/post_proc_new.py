import csv
from statistics import mean
import numpy as np
import os
import argparse
from sklearn.preprocessing import minmax_scale


def post_processing(val_path, evaluation_file, new_evaluation_file, n_shots=5):
    '''Post processing of a prediction file by removing all events that have shorter duration
    than 200 ms.
    
    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    '''
    

    results = []
    with open(evaluation_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            results.append(row)

    new_results = [['Audiofilename', 'Starttime', 'Endtime']]
    for event in results:
        audiofile = event[0]
        
        
        if float(event[2])-float(event[1]) >= 0.200:
            new_results.append(event)

    with open(new_evaluation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_results)
        
    return
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    experiment_result_path = "/home/tyz/data/data_experiment/DCASE_2022_few_shot/baseline_test_1"
    parser.add_argument('-val_path', type=str, help='path to validation folder with wav and csv files',default="/home/tyz/data/DCASE_2022_Few_shot_Bioacoustic_Event_Detection/Development_Set/Validation_Set")
    parser.add_argument('-evaluation_file', type=str, help='path and name of prediction file',default=os.path.join(experiment_result_path,"Eval_out.csv"))
    parser.add_argument('-new_evaluation_file', type=str, help="name of prost processed prediction file to be saved",default=os.path.join(experiment_result_path,"fixed_eval_out.csv"))
    
    args = parser.parse_args()

    post_processing(args.val_path, args.evaluation_file, args.new_evaluation_file)
