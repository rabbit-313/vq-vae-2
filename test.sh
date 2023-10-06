# python test.py [CHECKPOINT] [INPUT_Folder_PATH] [OUTPUT_Folder_PATH]

# INPUT_Folder_PATHにはフォルダが入った，フォルダのパスを指定してください
# Folder_A
#   Folder_B
#       1.png
#       2.png
#       3.png
#つまり，上記のような階層にし，Folder_Aのパスを指定してください
python test.py --checkpoint /home/ru/ドキュメント/study/SR/VAE/vq-vae-2/checkpoint/ColorLossGaussianTrue/vqvae_032.pt --dataset /home/ru/ドキュメント/study/SR/Datasets/DF2K_bicubic --savefolder /home/ru/ドキュメント/study/SR/VAE/vq-vae-2/SR_dataset/DIV2K_Flickr/ColorLossGausianTrue32
