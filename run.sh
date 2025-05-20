lr=2e-3
gradient_accumulation_steps=1
bs=8
orthogonal_loss_weight=0
norm_loss_weight=0
corr_loss_weight=0
load_balance_loss_weight=0.5
num_heads=10
data_path="700k"
loss_type=origin
num_train_epochs=1

loss_components=loss
if [ $orthogonal_loss_weight != 0 ]; then
    loss_components="${loss_components}_orthogonal${orthogonal_loss_weight}"
fi
if [ $norm_loss_weight != 0 ]; then
    loss_components="${loss_components}_norm${norm_loss_weight}"
fi
if [ $corr_loss_weight != 0 ]; then
    loss_components="${loss_components}_corr${corr_loss_weight}"
fi
if [ $load_balance_loss_weight != 0 ]; then
    loss_components="${loss_components}_loadBalance${load_balance_loss_weight}"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/config.yaml \
    --num_processes=8 --main_process_port=29506 --gradient_accumulation_steps=$gradient_accumulation_steps learn_sharebase.py \
    --learning_rate=$lr --loss_type=$loss_type --num_heads=$num_heads \
    --wandb_name=sharebase_${loss_type}_${data_path}_heads${num_heads}_lr${lr}_${loss_components}_epoch${num_train_epochs} --data_path=$data_path \
    --per_device_train_batch_size=$bs --num_train_epochs=$num_train_epochs \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --orthogonal_loss_weight=$orthogonal_loss_weight \
    --norm_loss_weight=$norm_loss_weight \
    --corr_loss_weight=$corr_loss_weight \
    --load_balance_loss_weight=$load_balance_loss_weight \
    --base_model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft" \
    | tee -a log/${loss_type}_${data_path}_heads${num_heads}_lr${lr}_${loss_components}.log # 2>&1