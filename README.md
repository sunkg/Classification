This repository contains code and part of the released data for the following manuscript:

Kaicong Sun, Yuanwang Zhang, Jiameng Liu, Han Zhang, Qian Wang, and Dinggang Shen, "A Unified Single-modal Framework for Brain Disease Classification Achieving Multi-modal Performance", 2023 

Prerequisites:
1. PyTorch (1.10 or greater)
2. cuda (11.0 or greater)
3. numpy (1.19 or greater)
4. Nibabel (3.2 or greater)
5. h5py (3.1 or greater)

Download the data examples and pretrained models from the following link:
https://pan.baidu.com/disk/main?_at_=1694942859828#/index?category=all&path=%2FNC_Data

Unzip these two downloaded folders within the root directory of these github files.
The pretrained models corresponding to the five used datasets can be found under directory ./Models
The partially released data can be found under directory ./Data_examples

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
To test the pretrained models, you can run the following scripts in command line:

###### Test Code: ADNI (NC vs. AD) ######
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/ADNI/NCAD --test_path Data_examples/ADNI/test_NCAD/Fold1 --table_csv Data_examples/ADNI/Table.csv --fold 1 --protocals CN AD 

###### Test Code: ADNI (sMCI vs. pMCI) ######
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/ADNI/sMCIpMCI --test_path Data_examples/ADNI/test_sMCIpMCI/Fold1 --table_csv Data_examples/ADNI/Table.csv --fold 1 --protocals sMCI pMCI 

###### Test Code: OASIS (NC vs. AD) ######
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/OASIS --test_path Data_examples/OASIS/test/Fold1 --table_csv Data_examples/OASIS/Table.csv --fold 1 --protocals CN AD --dataset OASIS --modalities AV45 MRI

###### Test Code: HS Hospital (NC vs. AD) ######
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/HSHospital --test_path Data_examples/HSHospital/test/Fold1 --table_csv Data_examples/HSHospital/Table.csv --fold 1 --protocals CN AD --dataset HS --modalities FDG AV45 MRI

###### Test Code: RJ Hospital (NCI vs. svMCI) ###### 
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/RJHospital --test_path Data_examples/RJHospital/test/Fold1 --table_csv Data_examples/RJHospital/Table.csv --fold 1 --protocals NCI a-MCI --dataset RJ --modalities Flair MRI

###### Test Code: BraTS 2021(MGMT+ vs. MGMT-) ###### 
python3 eval.py --test_mode J --save_path /path_to_save --model_path Models/BraTS2021 --test_path Data_examples/BraTS2021/test/Fold1 --table_csv Data_examples/BraTS2021/Table.csv --fold 1 --protocals 0 1 --dataset BraTS --modalities Flair T2 T1CE T1

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
To retrain the model, you can run the following scripts in command line:
 
###### Training Stage I for ADNI: NC vs. AD (3D CNN Backbone) ######
python3 train.py --train C --logdir  /path_to_save  --data_path /path_to_data  --table_csv /path_to_csv --batch_size 10  --protocals CN AD  --datablocksize 10 --lr 1e-4  

###### Training Stage II for ADNI: NC vs. AD (3D CNN Backbone) ###### 
python3 train.py --train J --logdir /path_to_save --Load_CNetwork /path_to_saved_model_from_Stage1 --data_path /path_to_data  --table_csv /path_to_csv --batch_size 12 --protocals CN AD --datablocksize 12 --lr 1e-3 

------------------------------------------------------------------------

###### Training Stage I for ADNI: sMCI vs. pMCI (3D CNN Backbone) ###### 
python3 train.py --train C --logdir /path_to_save  --data_path /path_to_data  --table_csv /path_to_csv --batch_size 8  --protocals sMCI pMCI --datablocksize 8 --lr 5e-5 --mod_augment

###### Training Stage II for ADNI: sMCI vs. pMCI (3D CNN Backbone) ######
python3 train.py --train J --logdir /path_to_save --Load_CNetwork /path_to_saved_model_from_Stage1 --data_path /path_to_data  --table_csv /path_to_csv --batch_size 6  --protocals sMCI pMCI --datablocksize 6 --lr 1e-3  --mod_augment

------------------------------------------------------------------------

###### Training Stage I for ADNI: NC vs. AD (MiSePyNet Backbone) ###### 
python3 train.py --train C --logdir  /path_to_save  --data_path /path_to_data  --table_csv /path_to_csv --batch_size 10  --protocals CN AD  --datablocksize 10 --lr 5e-5  --backbone PyramidCNN

###### Training Stage II for ADNI: NC vs. AD (MiSePyNet Backbone) ######
python3 train.py --train J --logdir /path_to_save --Load_CNetwork /path_to_saved_model_from_Stage1 --data_path /path_to_data  --table_csv /path_to_csv --batch_size 10 --protocals CN AD --datablocksize 10 --lr 5e-4 --backbone PyramidCNN

------------------------------------------------------------------------

###### Training Stage I for ADNI: sMCI vs. pMCI (MiSePyNet Backbone) ###### 
python3 train.py --train C --logdir  /path_to_save  --data_path /path_to_data  --table_csv /path_to_csv --batch_size 10 --protocals sMCI pMCI --datablocksize 10 --lr 1e-4 --backbone PyramidCNN --mod_augment

###### Training Stage II for ADNI: sMCI vs. pMCI (MiSePyNet Backbone) ###### 
python3 train.py --train J --logdir /path_to_save --Load_CNetwork /path_to_saved_model_from_Stage1 --data_path /path_to_data  --table_csv /path_to_csv --batch_size 16 --protocals sMCI pMCI  --datablocksize 16 --lr 1e-3 --backbone PyramidCNN --mod_augment

------------------------------------------------------------------------

###### Training Stage I for OASIS using warm-up by ADNI: NC vs. AD (3D CNN Backbone) ###### 
python3 train.py --train C --logdir /path_to_save --Load_CNetwork /path_to_saved_model_from_ADNI_Stage1 --table_csv /path_to_csv --data_path /path_to_data --batch_size 8  --protocals CN AD  --modalities AV45 MRI --datablocksize 8 --lr 5e-5 --lambda_epochs 15  --dataset OASIS 

###### Training Stage II for OASIS using warm-up by ADNI: NC vs. AD (3D CNN Backbone) ###### 
python3 train.py --train J --logdir /path_to_save --Load_JNetwork /path_to_saved_model_from_ADNI_Stage2  --Load_CNetwork /path_to_saved_model_from_Stage1 --data_path /path_to_data  --table_csv /path_to_csv --protocals CN AD --batch_size 4 --modalities AV45 MRI --datablocksize 4 --lr 5e-5 --weight_c 1e-4  --lambda_epochs 15 --dataset OASIS 

------------------------------------------------------------------------

###### Training Stage I for BraTS2021: MGMT+ vs. MGMT- (3D CNN Backbone) ###### 
python3 train.py --train C --logdir  /path_to_save --table_csv /path_to_csv --data_path /path_to_data --batch_size 6  --protocals 0 1 --modalities Flair T2 T1CE T1 --datablocksize 6 --lr 5e-6 --milestone 30 --lambda_epochs 30 --dataset BraTS 

###### Training Stage II for BraTS2021: MGMT+ vs. MGMT- (3D CNN Backbone) ###### 
python3 train.py --train J --logdir /path_to_save  --Load_CNetwork /path_to_saved_model_from_Stage1 --table_csv /path_to_csv --data_path /path_to_data --batch_size 4 --protocals 0 1 --modalities Flair T2 T1CE T1 --datablocksize 4 --lr 1e-4 --weight_c 2e-4 --lambda_epochs 15 --dataset BraTS 








