from freeEnergyTMP import TMPModel, SimpleWeightModel, TMPModelWithSimpleWeights
import time, itertools
from c3ddata_processing.data_processing import Data_processing
import torch
import os 
import numpy as np
import random
import matplotlib.pyplot as plt
from linear_operator.utils.errors import NanError

def getSeeds():
    seeds = {
        'random_seed': random.getstate()[1][0],
        'np_seed': np.random.get_state()[1][0],
        'torch_seed': torch.seed()
    }
    print(seeds)
    return {'seeds': seeds}

def get_VAF(time_point_list,MP_training_data_list,weight_dists,model):
    old_mode=model.training
    model.eval()
    residual=sum([(( dat.T - model(tp,weights=w).mean )**2).sum() for tp,dat,w in zip(time_point_list,MP_training_data_list,weight_dists)])
    all_data=torch.cat(MP_training_data_list,dim=1)
    model.train(old_mode)
    
    return float(1.0-residual/all_data.numel()/all_data.var(dim=1).mean().detach())

def get_mean_VAF(time_point_list,MP_training_data_list,weight_dists,model):
    old_mode=model.training
    model.eval()
    residual=sum([(( dat.T - model(tp,weights=w).mean )**2).sum(0) for tp,dat,w in zip(time_point_list,MP_training_data_list,weight_dists)])
    all_data=torch.cat(MP_training_data_list,dim=1)
    model.train(old_mode)
    
    return float((1.0-residual/all_data.shape[1]/all_data.var(dim=1).detach()).mean())

def inferWeights(time_point_list,training_data_list,model):
        """Infer weights for data in training_data_list, using the entries of time_point_list as inputs
        returns list of weight distributions, and inference time"""
        
        # compute weights for each segment
        start_time=time.time()
        model.training_mode(weights=True)
        model.weights.covariance_mode("full")
        weight_dists=[]
        for tp,data in zip(time_point_list,training_data_list): 
                
            model.zero_grad()
            loss=model.weight_loss(tp,data)
            loss.backward()
            model.weights.weight_update()
            weight_dists.append(model.weights.detached_posterior())
            loss_after=model.weight_loss(tp,data).detach()
            assert loss_after <loss*1.0001, "before {0:f}, after {1:f}".format(float(loss),float(loss_after))
                
        weight_time=time.time()-start_time
        return weight_dists,weight_time

def MP_weight_model(relative_path,single_folder_path,num_DFs,num_MPs,max_iter,lr,optim,epoch_range,verbose = False):
    all_seeds = getSeeds()
    
    if verbose == True: verboseprint = print
    else: 
        def verboseprint(*args,**kwargs): 
            pass
    use_cuda = torch.cuda.is_available() 

    print('Use cuda: ' +str(use_cuda))
    
    data_import = Data_processing(relative_path = relative_path,single_folder_path = single_folder_path)
    if single_folder_path is None:
        data_var_mean,MP_training_data_list,time_point_list,segmentslenperperson = data_import.get_all_c3d_files()
    else:
        data_var_mean,MP_training_data_list,time_point_list,segmentslenperperson = data_import.get_one_c3d_files()
    
    torch.save(MP_training_data_list, 'Test_data_list.pth')


    segment_weights = torch.ones(len(time_point_list))
    inducing_time_point = torch.arange(0.0,0.4,0.03)
    noise_level = data_var_mean * 0.05
    if noise_level.numel() > 1:
        getVAF = get_mean_VAF
    else:  
        getVAF = get_VAF

    # Init TMPModelWithSimpleWeights model
    MP_model = TMPModelWithSimpleWeights(num_DFs=num_DFs,num_MPs=num_MPs,inducing_time_points = inducing_time_point,noise_target = noise_level)
    MP_model.MPs.init_SVD(MP_training_data_list)

    # initial weights dist.
    weight_dists=[MP_model.weights.prior]*len(time_point_list)

    # weight update 
    SWELBO_old=-1e100

    MP_model.train()
    # MP_model.training_mode(weights=True)
    
    # MP_model.weights.covariance_mode("tril")
    VAF_List = []
    SWELBO_List = []

    convergence_criterion = 3.0

    em_iter = 0
    while em_iter < max_iter:
        
        try:
            SWELBO_new=float(MP_model.segment_weighted_ELBO(time_point_list,MP_training_data_list,weight_dists,segment_weights).detach())
            VAF=getVAF(time_point_list,MP_training_data_list,weight_dists,MP_model)
            print("Iteration",em_iter,"ELBO before weight update",SWELBO_new,"VAF=",VAF,"SWELBO_old",SWELBO_old)
            assert SWELBO_new > SWELBO_old
            current_time = time.time()
            current_time_str = str(int(current_time))
            model_path = os.path.join("rare_disease","model_state_all", f'model_MP{num_MPs}_{current_time_str}.pth')
            torch.save(MP_model.state_dict(),model_path)
            if em_iter > 3 and (SWELBO_new-SWELBO_before_EM_step) < convergence_criterion:
                print (f"""Reached convergence criterion: {convergence_criterion} for SWELBO improvement across one EM_step. 
                    SWELBO after minus SWELBO before {em_iter}'th EM step was {SWELBO_new-SWELBO_before_EM_step}.""")
                break
            em_iter += 1
        except (AssertionError, NanError) as error:
            print(error)
            MP_model = TMPModelWithSimpleWeights(num_DFs=num_DFs,num_MPs=num_MPs,inducing_time_points = inducing_time_point,noise_target = noise_level)
            MP_model.load_state_dict(torch.load(model_path))
            lr *= 0.5
            print ("""The SWELBO decreased during optimization step. Reset the model state to before last weight update. 
                   Learning rate has been changed to """,lr)
            print (optimizer)

            SWELBO_new=float(MP_model.segment_weighted_ELBO(time_point_list,MP_training_data_list,weight_dists,segment_weights).detach())
            VAF=getVAF(time_point_list,MP_training_data_list,weight_dists,MP_model)
            print(f'Recalculated SWELBO_new for reset model state. SWELBO_new is {SWELBO_new}, VAF is {VAF}. em_iter is {em_iter}.')

          
        SWELBO_old=SWELBO_new
        SWELBO_before_EM_step = SWELBO_new

        weight_dists,weight_time=inferWeights(time_point_list,MP_training_data_list,MP_model)

        verboseprint("weight update time",weight_time)

        SWELBO_new=float(MP_model.segment_weighted_ELBO(time_point_list,MP_training_data_list,weight_dists,segment_weights).detach())
        VAF=getVAF(time_point_list,MP_training_data_list,weight_dists,MP_model)
        
        
        MP_model.training_mode(mps=True,likelihood=(em_iter>10) or (VAF>0.95))
        
        
        print("ELBO after weight update, before MP update",SWELBO_new,"VAF=",VAF,"old ELBO",SWELBO_old)
        if not VAF_List or VAF != VAF_List[-1]:
            VAF_List.append(VAF)

        assert( (SWELBO_new-SWELBO_old) > -abs(SWELBO_old)*0.01 )
        SWELBO_old=SWELBO_new

        if not SWELBO_List or abs(SWELBO_new - SWELBO_List[-1]) > 1:
            SWELBO_List.append(SWELBO_new)
        
        # # preprare flattended data
        # if use_cuda: # on a GTX 960, MP updates are 6x faster on GPU
        #     MP_model=MP_model.cuda()
        #     print("Cuda_used")
        
            
        # unrolled_data=MP_model.unroll_data(time_point_list,MP_training_data_list, weight_dists,segment_weights)
        
        # if use_cuda:
        #     mem_avail=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        #     datapoint_size=sum([d.numel()*d.element_size() for d in unrolled_data[0]])
        #     max_batch_size=mem_avail/datapoint_size 
        
       # data_loader=torch.utils.data.DataLoader(unrolled_data,batch_size=4096,pin_memory=True)
        
        if optim == 1:
            optimizer = torch.optim.LBFGS(itertools.chain(MP_model.parameters()), lr)
        if optim == 2:
            optimizer = torch.optim.Adam(MP_model.parameters(), lr)

        last_loss=1e100
        start_time=time.time()
        for epoch in range(epoch_range): 
            try:
                def closure():
                    
                    optimizer.zero_grad()
                    #total_loss=torch.tensor(0.0)
                    mp_loss = MP_model.MP_loss(time_point_list,MP_training_data_list,weight_dists,segment_weights)
                    mp_loss.backward()

                    # for unrolled_inputs in data_loader:
                    #     loss=MP_model.unrolled_MP_loss(MP_model.unrolled_data_to_same_device(unrolled_inputs) )
                    #     loss.backward()
                    #     total_loss =total_loss + loss.detach()
                    
                    return mp_loss

                optimizer.step(closure)

                # termination check
                new_loss=closure().cpu().detach().numpy()
                print("at iteration",epoch,"loss is",new_loss)#,MP_model.likelihood.noise)
                if np.isnan(new_loss): break
                if (last_loss-new_loss)<0.1:
                    break
                last_loss=new_loss
            except NanError:
                break


        verboseprint("MP update time, tensor based learning",time.time()-start_time)
        # move model back to cpu for weight updates, which are 6x slower on the GPU
        MP_model=MP_model.cpu()
        
    
    # x_axis = range(1,len(VAF_List))
    # plt.plot(x_axis, VAF_List[1:], marker='o', linestyle='-', color='b', label='Data')
    # plt.show()
    # print("Training took",time.time()-start_time,"seconds")
    time_points= torch.arange(0.0,0.4,0.005)
    length_scale = MP_model.MPs.covar_module.base_kernel.lengthscale
    movement_primitives = MP_model.MPs(time_points)
    final_model_path = os.path.join("rare_disease","model_state_all", f'final_model_MP{num_MPs}.pth')
    torch.save(MP_model.state_dict(),final_model_path)

    return weight_dists,segmentslenperperson,VAF_List,SWELBO_List,movement_primitives,all_seeds,length_scale,final_model_path,lr



def get_weight(relative_path,num_DFs,num_MPs,max_iter):
    all_seeds = getSeeds()

    data_import = Data_processing(relative_path = relative_path,single_folder_path = single_folder_path)

    if single_folder_path is None:
        data_var_mean,MP_training_data_list,time_point_list,segmentslenperperson = data_import.get_all_c3d_files()
    else:
        data_var_mean,MP_training_data_list,time_point_list,segmentslenperperson = data_import.get_one_c3d_files()

    segment_weights = torch.ones(len(time_point_list))
    inducing_time_point = torch.arange(0.0,0.4,0.02)
    noise_level = data_var_mean * 0.05
    if noise_level.numel() > 1:
        getVAF = get_mean_VAF
    else:  
        getVAF = get_VAF

    # Init TMPModelWithSimpleWeights model
    MP_model_for_weights = TMPModelWithSimpleWeights(num_DFs=num_DFs,num_MPs=num_MPs,inducing_time_points = inducing_time_point,noise_target = noise_level)
    model_path_weight = os.path.join("rare_disease","model_state","final_model_MP16.pth")
    MP_model_for_weights.load_state_dict(torch.load(model_path_weight))

    # initial weights dist.
    weight_dists=[MP_model_for_weights.weights.prior]*len(time_point_list)

    # weight update 
    SWELBO_old=-1e100

    MP_model_for_weights.train()
    # MP_model_for_weights.training_mode(weights=True)
        
    SWELBO_before=float(MP_model_for_weights.segment_weighted_ELBO(time_point_list,MP_training_data_list,weight_dists,segment_weights).detach())
    VAF_before=getVAF(time_point_list,MP_training_data_list,weight_dists,MP_model_for_weights)

    weight_dists,weight_time=inferWeights(time_point_list,MP_training_data_list,MP_model_for_weights)
   
    SWELBO_after=float(MP_model_for_weights.segment_weighted_ELBO(time_point_list,MP_training_data_list,weight_dists,segment_weights).detach())
    VAF_after=getVAF(time_point_list,MP_training_data_list,weight_dists,MP_model_for_weights)

    return weight_dists,segmentslenperperson, SWELBO_before, SWELBO_after,VAF_after,VAF_before

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available() 
    print('Use cuda: ' +str(use_cuda))
    torch.set_default_tensor_type(torch.DoubleTensor)
    relative_path = "../Desktop/Kompletter_Datensatz/Kompletter_Datensatz/"

    current_directory = os.getcwd()
    

    # print("Current Directory:", current_directory)
    single_folder_path = None
    # single_folder_path = os.path.join(relative_path,"patients","021","Termin_1")
    # single_folder_path = os.path.join(relative_path,"Norm","001","linke_Seite")



    # results = []
    # training_time_list = []
    # for epoch_range in range (10,101,10):
    #     seed_state = random.getstate()
    #     time_1 = time.time()
    #     print(f"Epoch range  is {epoch_range}")
    #     result = MP_weight_model(relative_path=relative_path,
    #                             single_folder_path = single_folder_path,
    #                             num_DFs=42,
    #                             num_MPs=9,
    #                             max_iter=100,
    #                             lr=1, 
    #                             optim=1, 
    #                             epoch_range=epoch_range)
        
    #     weight_dists,segmentslenperperson,vaf_list, swelbo_list ,movement_primitives,all_seeds,length_scale,final_model_path,lr = result
    #     results.append((weight_dists,segmentslenperperson,epoch_range, vaf_list, swelbo_list,all_seeds,length_scale,lr))
    #     train_time = time.time() - time_1
    #     training_time_list.append(train_time)
    #     print(f'The training time now is {training_time_list}')
    
    # with open("HC_1EpochRange.txt", "w") as file:
    #     for _,_,epoch_range, vaf_list, swelbo_list, all_seeds,length_scale,lr in results:
    #         file.write(f"epoch_range = {epoch_range}\n")
    #         file.write(f"VAF_List: {vaf_list}\n")
    #         file.write(f"ELBO_List: {swelbo_list}\n")
    #         file.write(f"Length_Scale: {length_scale}\n")
    #         file.write(f"learning_rate: {lr}\n")
    #     file.write(f"training_time: {training_time_list}\n")

    # results_dict = dict ()

    # for _,_,epoch_range, vaf_list, swelbo_list,all_seeds,length_scale,lr in results:
    #     epoch_key = f"epoch_range_{epoch_range}"
    #     results_dict[epoch_key]={
    #         "vaf_list":vaf_list,
    #         "swelbo_list": swelbo_list,
    #         "all_seeds":all_seeds,
    #         "Length_Scale":length_scale,
    #         "lr":lr}

    
    # torch.save(results_dict, os.path.join('HC_1EpochRange.pkl')) 

    # plt.figure(figsize=(12, 6))
    # for _, _, epoch_range, vaf_list, swelbo_list,all_seeds,length_scale, lr in results:
    #     plt.subplot(1, 2, 1)
    #     x_range_vaf = range(3, len(vaf_list))
    #     plt.plot(x_range_vaf, vaf_list[3:], label=f"Epoch range = {epoch_range}")
    #     plt.title("VAF List")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("VAF Value")
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     x_range_swelbo = range(3, len(swelbo_list))
    #     plt.plot(x_range_swelbo, swelbo_list[3:], label=f"Epoch range = {epoch_range}")
    #     plt.title("ELBO List")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("ELBO Value")
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig("HC_1EpochRange.png")







    




    # result = get_weight(relative_path = relative_path,
    #                     num_DFs=42,
    #                     num_MPs = 16,
    #                     max_iter=100)
    
    # weight_dists,segmentslenperperson, SWELBO_before, SWELBO_after,VAF_after,VAF_before = result
    # print("VAF_new is:",VAF_after,"ELBO_new is:",SWELBO_after)

    # result_weight = {
    #     "Weights": {},
    #     "VAF":VAF_after,
    #     "ELBO":SWELBO_after
    # }

    # segment_index = 0 
    # data_import = Data_processing(relative_path = relative_path,single_folder_path = single_folder_path)
    # folder_path_list = data_import.get_norm_patients_folder()
    # for i, (length, datapath) in enumerate(zip(segmentslenperperson, folder_path_list), start=1):
    #     path_key = f"Path_{i}"
    #     result_weight["Weights"][path_key] = {}
    #     for segment in range(length):
    #         segment_key = f"Segment_{segment}"
    #         result_weight["Weights"][path_key][segment_key] = {
    #             "Weight_mean": weight_dists[segment_index].mean,
    #             "Weight_covariance": weight_dists[segment_index].covariance_matrix
    #         }
    #         segment_index += 1

    # torch.save(result_weight, os.path.join('results_weight_16.pkl')) 
    
    if single_folder_path == None:

        # for group fit 
        result= MP_weight_model(relative_path = relative_path,
                                single_folder_path = single_folder_path,
                                num_DFs = 42,
                                num_MPs = 15, 
                                max_iter = 100,
                                lr = 1,
                                optim=1,
                                epoch_range =100) 
        
        weight_dists,segmentslenperperson,vaf_list,swelbo_list,movement_primitives,all_seeds,length_scale,final_model_path = result
        print("VAF_List is:",vaf_list,"ELBO_List is:",swelbo_list)

        mps_mean = movement_primitives.mean
        mps_covariance = movement_primitives.covariance_matrix

        results_all_participants = {
        "MPs": {
            "MP_mean": mps_mean,
            "MP_covariance": mps_covariance
        },
        "Weights": {},
        "VAF_List": vaf_list,
        "SWELBO_List": swelbo_list,
        "Length_Scale": length_scale
        }

        segment_index = 0
        data_import = Data_processing(relative_path = relative_path,single_folder_path = single_folder_path)
        folder_path_list = data_import.get_norm_patients_folder()
        for i, (length, datapath) in enumerate(zip(segmentslenperperson, folder_path_list), start=1):
            path_key = f"Path_{i}"
            results_all_participants["Weights"][path_key] = {}
            for segment in range(length):
                segment_key = f"Segment_{segment}"
                results_all_participants["Weights"][path_key][segment_key] = {
                    "Weight_mean": weight_dists[segment_index].mean,
                    "Weight_covariance": weight_dists[segment_index].covariance_matrix
                }
                segment_index += 1

        torch.save(results_all_participants, os.path.join('results_for_all_MP8.pkl')) 
    
    # if single_folder_path != None:

    #     num_MPs_list = [6, 7, 8, 9, 10, 11, 12, 13,14]
    #     results = []
    #     time_1 = time.time()

    #     for num_MPs in num_MPs_list:
    #         seed_state = random.getstate()
    #         print(f"Running with num_MPs = {num_MPs}")
    #         result = MP_weight_model(relative_path=relative_path,
    #                                 single_folder_path = single_folder_path,
    #                                 num_DFs=42,
    #                                 num_MPs=num_MPs,
    #                                 max_iter=50,
    #                                 lr=1, 
    #                                 optim=1, 
    #                                 epoch_range=50)
        
    #         weight_dists,segmentslenperperson,vaf_list, swelbo_list ,movement_primitives,all_seeds,length_scale,final_model_path,lr = result
    #         results.append((weight_dists,segmentslenperperson,num_MPs, vaf_list, swelbo_list,all_seeds,length_scale,lr))
    #     train_time = time.time()-time_1
    #     print(f"The training took: {train_time}")
    #     with open("Pat021_14.txt", "w") as file:
    #         for _,_,num_MPs, vaf_list, swelbo_list, all_seeds,length_scale,lr in results:
    #             file.write(f"num_MPs = {num_MPs}\n")
    #             file.write(f"VAF_List: {vaf_list}\n")
    #             file.write(f"ELBO_List: {swelbo_list}\n")
    #             file.write(f"Length_Scale: {length_scale}\n")
    #             file.write(f"training_time: {train_time}\n")
    #             file.write(f"learning_rate: {lr}\n")
        
    #     # Plotting

    #     results_dict = dict ()

    #     for _,_,num_MPs, vaf_list, swelbo_list,all_seeds,length_scale,lr in results:
    #         num_MPs_key = f"num_MPs_{num_MPs}"
    #         results_dict[num_MPs_key]={
    #             "vaf_list":vaf_list,
    #             "swelbo_list": swelbo_list,
    #             "all_seeds":all_seeds,
    #             "Length_Scale":length_scale}

        
    #     torch.save(results_dict, os.path.join('Pat021_14.pkl')) 

    #     plt.figure(figsize=(12, 6))
    #     for _, _, num_MPs, vaf_list, swelbo_list,all_seeds,length_scale, lr in results:
    #         plt.subplot(1, 2, 1)
    #         x_range_vaf = range(3, len(vaf_list))
    #         plt.plot(x_range_vaf, vaf_list[3:], label=f"num_MPs = {num_MPs}")
    #         plt.title("VAF List")
    #         plt.xlabel("Iteration")
    #         plt.ylabel("VAF Value")
    #         plt.legend()

    #         plt.subplot(1, 2, 2)
    #         x_range_swelbo = range(3, len(swelbo_list))
    #         plt.plot(x_range_swelbo, swelbo_list[3:], label=f"num_MPs = {num_MPs}")
    #         plt.title("ELBO List")
    #         plt.xlabel("Iteration")
    #         plt.ylabel("ELBO Value")
    #         plt.legend()

    #     plt.tight_layout()
    #     plt.savefig("Pat021_14.png")


