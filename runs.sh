#----------------------------------------------------------------------------------------------------------------------------------
# CIFAR 10
#----------------------------------------------------------------------------------------------------------------------------------
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_0_1' --drop_prob 0.1 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_0_2' --drop_prob 0.2 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_0_3' --drop_prob 0.3 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_0_4' --drop_prob 0.4 --layer_no 0

# Layer 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_4_1' --drop_prob 0.1 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_4_2' --drop_prob 0.2 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_4_3' --drop_prob 0.3 --layer_no 4 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_4_4' --drop_prob 0.4 --layer_no 4 --mode 'eval' --load_model True

# Layer 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_8_1' --drop_prob 0.1 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_8_2' --drop_prob 0.2 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_8_3' --drop_prob 0.3 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_8_4' --drop_prob 0.4 --layer_no 8

# Layer 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_12_1' --drop_prob 0.1 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_12_2' --drop_prob 0.2 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_12_3' --drop_prob 0.3 --layer_no 12 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_12_4' --drop_prob 0.4 --layer_no 12 --mode 'eval' --load_model True

# Layer 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_16_1' --drop_prob 0.1 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_16_2' --drop_prob 0.2 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_16_3' --drop_prob 0.3 --layer_no 16 --mode 'eval' --load_model True
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist' --model 'mcdc_simplenet' --run_name 'simplenet_16_4' --drop_prob 0.4 --layer_no 16 --mode 'eval' --load_model True
#----------------------------------------------------------------------------------------------------------------------------------
# CIFAR 100
#----------------------------------------------------------------------------------------------------------------------------------
# Baselines
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_0_1' --drop_prob 0.1 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_0_2' --drop_prob 0.2 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_0_3' --drop_prob 0.3 --layer_no 0
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_0_4' --drop_prob 0.4 --layer_no 0

# Layer 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_2_1' --drop_prob 0.1 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_2_2' --drop_prob 0.2 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_2_3' --drop_prob 0.3 --layer_no 4
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_2_4' --drop_prob 0.4 --layer_no 4

# Layer 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_8_1' --drop_prob 0.1 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_8_2' --drop_prob 0.2 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_8_3' --drop_prob 0.3 --layer_no 8
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_8_4' --drop_prob 0.4 --layer_no 8

# Layer 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_12_1' --drop_prob 0.1 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_12_2' --drop_prob 0.2 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_12_3' --drop_prob 0.3 --layer_no 12
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_12_4' --drop_prob 0.4 --layer_no 12

# Layer 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_16_1' --drop_prob 0.1 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_16_2' --drop_prob 0.2 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_16_3' --drop_prob 0.3 --layer_no 16
python MCDropConnect-PyTorch/main.py --use_colab True --dataset 'fmnist0' --model 'mcdc_simplenet' --run_name 'simplenet_16_4' --drop_prob 0.4 --layer_no 16
