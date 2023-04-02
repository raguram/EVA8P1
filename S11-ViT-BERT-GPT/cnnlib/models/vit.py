import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary

class PatchEmbedding(nn.Module): 
    
    def __init__(self, in_channels, patch_size, hidden_dim): 
    
        super(PatchEmbedding, self).__init__()

        ## Patcher 
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        ## Flatten 
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        
        x = self.patcher(x)
        return self.flatten(x).permute(0, 2, 1)

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, 
                 hidden_dim = 768, 
                 nheads = 12, 
                 attn_dropout = 0): 
        
        super(MultiHeadAttentionBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nheads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):

        x = self.layer_norm(x)
        # print(f'Shape of the input layer to multihead attn: {x.shape}')
        x, _ = self.multihead_attn(query = x, key = x, value = x, need_weights = False)
        return x 

class MLPBlock(nn.Module):

    def __init__(self, hidden_dim=768, mlp_dim=3072, dropout=0.1):

        super(MLPBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):

    def __init__(self, hidden_dim=768, mlp_dim=3072, nheads=12, attn_dropout=0, mlp_droput=0.1):

        super(TransformerEncoderBlock, self).__init__()

        self.msa_block = MultiHeadAttentionBlock(hidden_dim, nheads, attn_dropout)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, mlp_droput)

    def forward(self, x): 
        
        # print(f'Input to transformer encoder: {x.shape}')
        x = self.msa_block(x) + x

        # print(f'output of msa_block: {x.shape}')
        x = self.mlp(x) + x
        return x


class ViT(nn.Module):

    def __init__(self, 
                 img_size:int = 224,
                 in_channels:int = 3, 
                 patch_size: int = 16,
                 hidden_dim=768, 
                 mlp_dim=3072,
                 number_transformer_layers = 12,
                 nheads = 12,
                 out_classes=10):

        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, hidden_dim)

        # ## Add the cls token 
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim), requires_grad=True)

        ## Add the position embedding 
        number_of_patches = (img_size // patch_size) ** 2
        # print(f'Number of patches = {number_of_patches}')
        self.pos_embedding = nn.Parameter(torch.randn(1, number_of_patches + 1, hidden_dim), requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=0.1)

        # Add the transformer block 
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(hidden_dim, mlp_dim, nheads) for _ in range(number_transformer_layers)])

        # Add the classifier head 
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.Linear(hidden_dim, out_classes)
        )
            
    def forward(self, x): 

        batch_size = x.shape[0]
        # print(f'Batch_size: {batch_size}')

        ## Input is 3 x 224 x 224 
        x = self.patch_embedding(x)
        # print(f'Output after patch_embeddeing: {x.shape}')

        ## Concatenate the cls token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # print(f'Class token shape: {cls_token.shape}')

        x = torch.concat((cls_token, x), dim=1)
        # print(f'Concatenated the cls token: {x.shape}')

        ## Add the position embedding
        x = x + self.pos_embedding.expand(batch_size, -1, -1)
        x = self.embedding_dropout(x)
        # print(f'Added positional embedding: {x.shape}')
        
        # Now the size of the out is (batch_size, number_of_patches + 1, hidden_dim) - (512, 197, 768)
        x = self.transformer_encoder(x)
        # print(f'Transformer encoding: {x.shape}')
        # print(f'{x[:, 0].shape}')
        # Now the size of the out is (batch_size, number_of_patches + 1, hidden_dim) - (512, 197, 768)
        x = self.classifier(x[:, 0])

        print(f'{x.shape}')
        return x

    def summarize(self, input): 
        summary(self, input_size=input)

def main(): 
    net = ViT()

    net.to(getDevice()).summarize((32, 3, 224, 224))

def getDevice(): 
  return torch.device("cuda" if isCuda() else "cpu")

def isCuda():
  return torch.cuda.is_available()

if __name__ == "__main__": 
    main()
