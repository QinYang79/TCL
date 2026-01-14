data_name=f30k_precomp
data_path=/media/hdd1/hupeng/ImageTextMatch/SCAN/data/data
vocab_path=/home/qinyang/projects/RECO/vocab

dis_k=3
tau=0.05  
active=Exp
# ReLU|Exp
method=VSE_infty
#VSE|VSE_bert|VSE_glove|VSE_infty|VSE_infty_glove|VSE_infty_bert|SAF|SGR
gpu=0

folder_name=./runs/${data_name}_${method}_dis_k${dis_k}_tau${tau}_active${active} 

CUDA_VISIBLE_DEVICES=$gpu python train.py --method $method --gpu $gpu --dis_k $dis_k --tau $tau --active $active \
    --logger_path ${folder_name}/log --model_path ${folder_name}/checkpoint --dis_k $dis_k --data_name $data_name \
    --vocab_path $vocab_path --data_path $data_path --batch_size 128  --num_epochs 25 --lr_update 15 