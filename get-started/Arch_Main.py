import torch
from mmvae.trainers.Arch_Trainer import VAETrainer

configurations = {
    "xlarge": [8192, 2048, 512],
    "large": [4096, 1200, 400]
}

def main():

    batch_size = 32

    for name, sizes in configurations.items(): 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("Device:", device)
        #trainer = VAETrainer(device, sizes)
        #trainer.train()
        print(f'name: {name}, size: {sizes}')



if __name__ == "__main__":
    main()
