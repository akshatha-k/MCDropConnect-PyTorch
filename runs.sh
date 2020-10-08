#----------------------------------------------------------------------------------------------------------------------------------
# CIFAR 10
#----------------------------------------------------------------------------------------------------------------------------------
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_0_1' --drop_prob 0.1 --layer_no 0 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_0_2' --drop_prob 0.2 --layer_no 0 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_0_3' --drop_prob 0.3 --layer_no 0 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_0_4' --drop_prob 0.4 --layer_no 0 --mode 'eval' --load_model True

# Layer 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_4_1' --drop_prob 0.1 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_4_2' --drop_prob 0.2 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_4_3' --drop_prob 0.3 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_4_4' --drop_prob 0.4 --layer_no 4 --mode 'eval' --load_model True

# Layer 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_8_1' --drop_prob 0.1 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_8_2' --drop_prob 0.2 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_8_3' --drop_prob 0.3 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_8_4' --drop_prob 0.4 --layer_no 8

# Layer 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_12_1' --drop_prob 0.1 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_12_2' --drop_prob 0.2 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_12_3' --drop_prob 0.3 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_12_4' --drop_prob 0.4 --layer_no 12 --mode 'eval' --load_model True

# Layer 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_16_1' --drop_prob 0.1 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_16_2' --drop_prob 0.2 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_16_3' --drop_prob 0.3 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar10' --model 'mcdc_resnet20' --run_name 'resnet20_16_4' --drop_prob 0.4 --layer_no 16 --mode 'eval' --load_model True
#----------------------------------------------------------------------------------------------------------------------------------
# CIFAR 100
#----------------------------------------------------------------------------------------------------------------------------------
# Baselines
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_0_1' --drop_prob 0.1 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_0_2' --drop_prob 0.2 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_0_3' --drop_prob 0.3 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_0_4' --drop_prob 0.4 --layer_no 0

# Layer 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_2_1' --drop_prob 0.1 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_2_2' --drop_prob 0.2 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_2_3' --drop_prob 0.3 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_2_4' --drop_prob 0.4 --layer_no 4

# Layer 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_8_1' --drop_prob 0.1 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_8_2' --drop_prob 0.2 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_8_3' --drop_prob 0.3 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_8_4' --drop_prob 0.4 --layer_no 8

# Layer 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_12_1' --drop_prob 0.1 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_12_2' --drop_prob 0.2 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_12_3' --drop_prob 0.3 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_12_4' --drop_prob 0.4 --layer_no 12

# Layer 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_16_1' --drop_prob 0.1 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_16_2' --drop_prob 0.2 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_16_3' --drop_prob 0.3 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'cifar100' --model 'mcdc_resnet20' --run_name 'resnet20_16_4' --drop_prob 0.4 --layer_no 16
