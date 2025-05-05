import os
import torch
import glob
from c3ddata_processing.c3d import Reader
import numpy as np
from c3ddata_processing.c3d_metadata import print_metadata
from c3ddata_processing.c3d_label_left_right import get_label
import matplotlib.pyplot as plt

class Data_processing:

    def __init__(self,relative_path,single_folder_path = None):
        self.relative_path = relative_path
        self.single_folder_path = single_folder_path

    def get_norm_patients_folder(self):
        norm_folder = "holdoutNorm"
        patient_folder = "holdoutPatients"
        # norm_folder = "Norm"
        # patient_folder = "patients"
        subfolder_names_norm = sorted([name for name in os.listdir(os.path.join(self.relative_path, norm_folder)) if os.path.isdir(os.path.join(self.relative_path, norm_folder, name))],key= lambda x:int(x))
        subfolder_names_patient = sorted([name for name in os.listdir(os.path.join(self.relative_path, patient_folder)) if os.path.isdir(os.path.join(self.relative_path, patient_folder, name))])
        folder_path_list_norm = [os.path.join(self.relative_path, norm_folder, name, 'linke_Seite') for name in subfolder_names_norm]
        folder_path_list_patient = [os.path.join(self.relative_path, patient_folder, name, 'Termin_1') for name in subfolder_names_patient]
        folder_path_list = folder_path_list_norm + folder_path_list_patient
        return folder_path_list
    
    def get_label_contexts(self,file_path):
        with open(file_path,'rb') as handle:
            Icon,label_time,contexts  = get_label(Reader(handle))
        label_dict = dict ()
        for i, j,k in zip (Icon,label_time,contexts):
            j = self.custom_round(j)
            if i == True:
                label_dict[j] = k
        self.label_dict = dict(sorted(label_dict.items()))
        return self.label_dict
    
    def custom_round(self,value):
        value = round(value,3)
        remainder = round(value % 0.005,3) 

        if 0.001 <= remainder <= 0.0028:
            return value - remainder
        elif 0.0028 <= remainder <= 0.0041:
            return round((value + (0.005 -remainder)),3)
        else:
            return round(value, 3)
    
    def read_c3d(self,input_file): # get non edit c3d data from one clip, and remove the data we do not need
        with open(input_file,"rb") as handle:
            # print(input_file)
            reader = Reader(handle)
            start_index, finish_index, another_index = print_metadata(reader)
            data_list = []
            column = []
            # deal with two different sorted dof. 
            if finish_index - start_index == 15:
                for i, points, _ in reader.read_frames():
                    column.append(i)
                    selected_points = points[start_index: finish_index+1]
                    data_list.append(selected_points)
            
                data_array = np.array(data_list)
                column_array = np.array(column)/200
                data = np.array(data_array[:, :, :3])
                
                reshaped_data = data.reshape(data.shape[0], -1)
                transposed_data = np.transpose(reshaped_data)
                remove_data = [9,10,11,21,22,23] 

            else:
                for i, points, _ in reader.read_frames():
                    column.append(i)
                    selected_points = points[start_index: another_index+1]
                    data_list.append(selected_points)
                
                data_array = np.array(data_list)
                column_array = np.array(column)/200 
                data = np.array(data_array[:, :, :3])

                reshaped_data = data.reshape(data.shape[0], -1)
                transposed_data = np.transpose(reshaped_data)
                temp1 = transposed_data[36:]
                new_range = [temp1[3:6], temp1[:3], temp1[9:], temp1[6:9]]
                new_range_data = np.concatenate(new_range,axis = 0)
    
                transposed_data[36:] = transposed_data[24:36]
                transposed_data[24:36] = new_range_data
                remove_data = [6,7,8,21,22,23] 
            
            final_data = np.delete(transposed_data,remove_data, axis=0)
            self.data_timeheader = np.vstack((column_array,final_data))

        return self.data_timeheader  
    
    def process_c3d_files(self,folder_path):
        c3d_files = glob.glob(os.path.join(folder_path, "*.c3d"))
        c3d_files = sorted(c3d_files)
        self.data_outputs = []
        self.label_infos =[]

        for file_path in c3d_files:
            data_output = self.read_c3d(file_path)
            label_1Clip = self.get_label_contexts(file_path)
            self.data_outputs.append(data_output)
            self.label_infos.append(label_1Clip)
            # self.label_infos here will be a list of dictionary it might look like this 
            # [{1.665: 'left', 2.455: 'left', 2.05: 'right', 2.87: 'right'},
            # {2.42: 'left', 2.805: 'right', 3.17: 'left', 3.54: 'right'},
            # {3.075: 'left', 3.535: 'right', 3.985: 'left', 4.45: 'right'}]
        return self.label_infos,self.data_outputs
    
    def turn_data_left_right(self,oneclipdata):

        new_order = torch.tensor([0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,22,23,24,19,20,21,28,29,30,25,26,27,34,35,36,31,32,33,40,41,42,37,38,39])
        
        self.new_data = torch.index_select(torch.tensor(oneclipdata),dim = 0, index = new_order)

        return self.new_data
    
    def preparation1clip_left(self,label, data): # 
        time_in_clip = data[0]
        indexes = []

        # for key, value in label.items():
        #     index = np.where(np.isclose(time_in_clip,key,atol=0.001))
        #     indexes.append(index)

        for key,value in label.items():
            if value == 'left':
                index = np.where(np.isclose(time_in_clip,key,atol=0.001))
                if key > time_in_clip[-1]:
                    print('Note: In this clip, a label is out of the data range, set the last data index to the Index')
                    index = (np.array(len(time_in_clip)-1)),
                indexes.append(index)
        
        num_segments = len(indexes) - 1
        print('There are/is',num_segments,'in the clip')

        self.time_point_list_1clip = []
        self.MP_training_data_list_1clip = []

        for i in range(num_segments):
            start_index = int(indexes[i][0])
            end_index = int(indexes[i+1][0])       
            data_segment_with_time = data[:, start_index:end_index]
            time_points = data_segment_with_time[0] - data_segment_with_time[0][0]
            data_segment = np.radians(data_segment_with_time)[1:]

            scaled_time_points = []
            ratio= len(time_points)/80
            for timepoint in time_points:
                timepoint /= ratio
                scaled_time_points.append(timepoint)
            
            self.time_point_list_1clip.append(scaled_time_points)
            self.MP_training_data_list_1clip.append(data_segment)
        
        return self.MP_training_data_list_1clip, self.time_point_list_1clip
        
    def datapreparation_single(self,label_infos,data_outputs):
        MP_training_data_list_1person = []
        time_point_list_1person = []

        for label, data_output in zip(label_infos, data_outputs):
            MP_training_data_list_1clip, time_point_list_1clip = self.preparation1clip_left(label,data_output)
            MP_training_data_list_1person.append(MP_training_data_list_1clip)
            time_point_list_1person.append(time_point_list_1clip)
        self.MP_training_data_list_1person = [torch.tensor(arr) for sublist in MP_training_data_list_1person for arr in sublist]
        self.time_point_list_1person = [torch.tensor(arr).clone().detach() for sublist in time_point_list_1person for arr in sublist]

        return self.MP_training_data_list_1person,self.time_point_list_1person
    
    def get_one_c3d_files(self):
        MP_training_data_list = []
        time_point_list = []
        self.segmentslenperperson = []

        label_infos,data_outputs = self.process_c3d_files(self.single_folder_path)
        folder_to_check = os.path.join(self.single_folder_path,"re_nicht_betroffen")
        if os.path.isdir(folder_to_check) and folder_to_check.startswith(self.single_folder_path):
            new_data_outputs = []
            new_label_infos = []
            for label_1clip,oneclipdata in zip(label_infos,data_outputs):
                new_data = self.turn_data_left_right(oneclipdata)
                new_data_outputs.append(new_data)
                for key, value in label_1clip.items():
                    if value == 'left':
                        label_1clip[key] = 'right'
                    elif value == 'right':
                        label_1clip[key] = 'left'
                    else:
                        print("Aha, no way, that's impossible!")
                new_label_infos.append(label_1clip)
            MP_training_data_list_1person,time_point_list_1person = self.datapreparation_single(new_label_infos,new_data_outputs)
        else:
            MP_training_data_list_1person,time_point_list_1person = self.datapreparation_single(label_infos,data_outputs)

        MP_training_data_list.append(MP_training_data_list_1person)
        time_point_list.append(time_point_list_1person)
        self.segmentslenperperson.append(len(MP_training_data_list_1person))
        
        self.MP_training_data_list = [torch.tensor(arr) for sublist in MP_training_data_list for arr in sublist]
        self.time_point_list = [torch.tensor(arr) for sublist in time_point_list for arr in sublist]
        
        data_var = [] 
        for segment in self.MP_training_data_list:
            data_var.append(segment.var(axis=1))
        self.data_var_mean = torch.stack(data_var).mean(0)
        assert torch.atleast_1d(self.data_var_mean).shape == torch.Size([42]),f'We need one data variance value per degree of freedom. data_var_mean.shape was {torch.atleast_1d(self.data_var_mean).shape}'  # Calculate the mean of data_var list          

        return self.data_var_mean,self.MP_training_data_list,self.time_point_list,self.segmentslenperperson
    
    def check_single_data(self):
        var_mean,data_list,time_points,seg_len = self.get_one_c3d_files()
        index_list= [18,19,20,21,22,23]
        dofs_group = [data_list[0][i] for i in index_list]

        for dof in dofs_group:
            plt.plot(time_points[0],dof,label='DoF_Spine')
        plt.xlabel('Time')
        plt.legend(['X', 'Y', 'Z','X2','Y2','Z2']) 
        
        plt.savefig('plotSpine14.png')
        print('the plot has been saved')
         # Add legend labels if you have multiple curves
  

#         folder_to_check = os.path.join(self.single_folder_path,"re_betroffen")
#         if os.path.isdir(folder_to_check) and folder_to_check.startswith(self.single_folder_path):
#             time_points = data_outputs[0][0]
#             dofs_group_1 = data_outputs[0][1:4]
#             dofs_group_2 = data_outputs[0][10:13]
#             for dof in dofs_group_1:
#                 plt.plot(time_points, dof, label='Dof 1-3')
#             for dof in dofs_group_2:
#                 plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Comparison of Two Sets of Curves')
# plt.legend()


    def get_all_c3d_files(self): # put all data together, 
        MP_training_data_list = []
        time_point_list = []
        self.segmentslenperperson = []
        self.folder_path_list = self.get_norm_patients_folder()

        for folder_path in self.folder_path_list: # data of 75 participants 
            print(folder_path)
            label_infos,data_outputs = self.process_c3d_files(folder_path)
            check_affected_side = os.path.join(folder_path,"re_nicht_betroffen")
            if os.path.isdir(check_affected_side):
                new_data_outputs = []
                new_label_infos = []
                for label_1clip,oneclipdata in zip(label_infos,data_outputs):
                    new_data = self.turn_data_left_right(oneclipdata)
                    new_data_outputs.append(new_data)
                    for key, value in label_1clip.items():
                        if value == 'left':
                            label_1clip[key] = 'right'
                        elif value == 'right':
                            label_1clip[key] = 'left'
                        else:
                            print("Aha, no way, that's impossible!")
                    new_label_infos.append(label_1clip)
                MP_training_data_list_1person,time_point_list_1person = self.datapreparation_single(new_label_infos,new_data_outputs)

            else:
                MP_training_data_list_1person,time_point_list_1person = self.datapreparation_single(label_infos,data_outputs)
            print(f"There are {len(MP_training_data_list_1person)} segment/s from this participent")
            MP_training_data_list.append(MP_training_data_list_1person)
            time_point_list.append(time_point_list_1person)
            self.segmentslenperperson.append(len(MP_training_data_list_1person))

        
        self.MP_training_data_list = [torch.tensor(arr) for sublist in MP_training_data_list for arr in sublist]
        self.time_point_list = [torch.tensor(arr) for sublist in time_point_list for arr in sublist]
        data_var = [] 
        for segment in self.MP_training_data_list:
            data_var.append(segment.var(axis=1))
        self.data_var_mean = torch.stack(data_var).mean(0)
        assert torch.atleast_1d(self.data_var_mean).shape == torch.Size([42]),f'We need one data variance value per degree of freedom. data_var_mean.shape was {torch.atleast_1d(self.data_var_mean).shape}'  # Calculate the mean of data_var list          
        print(f'There are {len(self.MP_training_data_list)} Segments from all people')
        return self.data_var_mean,self.MP_training_data_list,self.time_point_list,self.segmentslenperperson






# if __name__ == "__main__":
#     relative_path = "../Desktop/Kompletter_Datensatz/"
#     single_folder_path = os.path.join(relative_path,"patients","162_Trochanterapophyseodese","Termin_1")
#     lookatthedata = Data_processing(relative_path = relative_path,right_affected_patients = right_affected_patients,single_folder_path=single_folder_path)

#     lookatthedata.check_single_data()

