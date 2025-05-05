import c3d
import glob
import os
from c3d_label_left_right import get_label

def get_c3d_file(folder_path_list):
    all_file_path = []
    all_segments_counter = 0 
    path_segment = dict() 
    for folder_path,i in zip(folder_path_list,range(1,len(folder_path_list)+1)):
        path_key = f'Path_{i}'   
        c3d_files = glob.glob(os.path.join(folder_path, "*.c3d"))
        c3d_files = sorted(c3d_files)
        segment_counter = 0 
        segment_contexts_list = []
        for file_path in c3d_files:
            all_file_path.append(file_path)
            label_dict = get_label_contexts(file_path)
            segment_counter += (len(label_dict)-1)
            n = len(label_dict) - 1
            segment_context = list(label_dict.values())[:n]
            segment_contexts_list.append(segment_context)
        print ("the path is",path_key,'it has',segment_counter,'segments')
        all_segments_counter += segment_counter
        boolean_list = [value == 'left' for sublist in segment_contexts_list for value in sublist]
        path_segment[path_key] = boolean_list
    print ('There are',all_segments_counter,'Segments')
    return path_segment

def get_label_contexts(file_path):
    with open(file_path,'rb') as handle:
        Icon,label_time,contexts  = get_label(c3d.Reader(handle))
    label_dict = dict ()
    for i, j,k in zip (Icon,label_time,contexts):
        if i == True:
            label_dict[j] = k
    label_dict = dict(sorted(label_dict.items()))
    return label_dict
            


# def get_label(file_path):
#     with open(file_path,'rb') as handle:
#         results = print_label(c3d.Reader(handle))
#     icon = results[0][0]
#     label_time = results[1][1]
#     strike= []
#     for i, j in zip (icon,label_time):
#         if i == True:
#             strike.append(j)
#     non_zero_values = [value for row in strike for value in row if value != 0.0]
#     rounded_values = [custom_round(value) for value in non_zero_values]
#     label = sorted (rounded_values)
#     return label    

if __name__ == "__main__":
    relative_path = "../Kompletter_Datensatz/"
    norm_folder = "Norm"
    patient_folder = "patients"
    subfolder_names_norm = sorted([name for name in os.listdir(os.path.join(relative_path, norm_folder)) if os.path.isdir(os.path.join(relative_path, norm_folder, name))],key= lambda x:int(x))
    subfolder_names_patient = sorted([name for name in os.listdir(os.path.join(relative_path, patient_folder)) if os.path.isdir(os.path.join(relative_path, patient_folder, name))])
    folder_path_list_norm = [os.path.join(relative_path, norm_folder, name, 'linke_Seite') for name in subfolder_names_norm]
    folder_path_list_patient = [os.path.join(relative_path, patient_folder, name, 'Termin_1') for name in subfolder_names_patient]
    folder_path_list = folder_path_list_norm + folder_path_list_patient
    
    c3d_files = get_c3d_file(folder_path_list)

