# =============================================================================
# code to generate tables from results\ .txt files
# =============================================================================

import numpy as np

# for m in (["EDOF_CNN_max","EDOF_CNN_3D","EDOF_CNN_fast","EDOF_CNN_pairwise","EDOF_CNN_pack_02","EDOF_CNN_pack"]):
# for m in (['EDOF_CNN_pack_38','EDOF_CNN_pack_37','EDOF_CNN_pack_36','EDOF_CNN_pack_35','EDOF_CNN_pack_34']):
# for m in (['EDOF_CNN_pack_41','EDOF_CNN_pack_40','EDOF_CNN_pack_39']):
# for m in (['EDOF_CNN_pack_32','EDOF_CNN_pack_33']):
for m in (["EDOF_CNN_pack_43"]):
# for m in (["EDOF_CNN_fast","EDOF_CNN_max"]):
# for m in (["EDOF_CNN_fast"]):
    # for s in ([3, 5]):
    for s in ([5]):
        mse=[]
        rmse=[]
        ssim=[]
        pnsr=[]
        for i in range(5):
            text_file = open("my_results_fixed_3/dataset-cervix93_zstacks-image_size-512-method-"+str(m)+"-Z-"+str(s)+"-fold-"+str(i)+"-epochs-200-batchsize-1-lr-0.001-cudan-0-image_channels-3channels-automate-1-augmentation-0-augmentLevel-0-rotate-0-hflip-0-ssim-0-jitter-0-comment-.txt", "r")
            
           
            lines = text_file.readlines()
            mse.append(float(lines[3].split(":")[1]))
            rmse.append(float(lines[4].split(":")[1]))
            ssim.append(float(lines[5].split(":")[1]))
            pnsr.append(float(lines[6].split(":")[1]))
           
        msem=np.mean(mse)
        rmsem=np.mean(rmse)
        ssimm=np.mean(ssim)
        pnsrm=np.mean(pnsr)
        msed=np.std(mse)
        rmsed=np.std(rmse)
        ssimd=np.std(ssim)
        pnsrd=np.std(pnsr)
        
        
        # print("&& "+str(s)+" & $"+str(round(msem,5))+" \pm "+ str(round(msed, 5))+"$ & $"+
        #       str(round(rmsem,5))+" \pm "+ str(round(rmsed, 5))+"$ & $"+
        #       str(round(ssimm*100,3))+" \pm "+ str(round(ssimd*100, 3))+"$ & $"+
        #       str(round(pnsrm,3))+" \pm "+ str(round(pnsrd, 3))+"$")
        print(str(m)+" mse: "+str(round(msem,5))+"+-"+ str(round(msed, 5))+" rmse: "+
              str(round(rmsem,5))+"+-"+ str(round(rmsed, 5))+" ssim: "+
              str(round(ssimm*100,3))+"+-"+ str(round(ssimd*100, 3))+" pnsrn: "+
              str(round(pnsrm,3))+"+-"+ str(round(pnsrd, 3))+"\n")
