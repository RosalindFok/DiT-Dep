import torch
import torch.nn as nn
from dataclasses import dataclass

from config import Experiment_Config

device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class Model_Returns:
    logits : torch.Tensor
    x_proj : torch.Tensor
    x_pred : torch.Tensor

class Time_Embedding(nn.Module):
    def __init__(self, dim : int, timesteps : int) -> None:
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim//2, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=dim//2, out_features=dim, bias=True)
        )

    def forward(self, t : torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1) # (B, 1)
        t = t * torch.exp(
            -torch.arange(0, self.dim//2, dtype=torch.float32, device=t.device) * (torch.log(torch.tensor(self.timesteps, dtype=torch.float32, device=t.device)) / (self.dim//2 - 1))
        ) # (B, dim//2)
        if self.dim % 2 == 0:
            t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1) # (B, dim)
        else: 
            t = torch.cat([torch.sin(t), torch.zeros_like(t[:,:1]), torch.cos(t)], dim=-1) # (B, dim)
        t = self.mlp(t).unsqueeze(1) # (B, 1, dim)
        return t

class Denoising_Network(nn.Module):
    def __init__(self, dim : int, use_att : bool, nhead : int = 8) -> None:
        super().__init__()
        assert dim % nhead == 0, f"dim={dim} must be divisible by nhead={nhead}"
        self.QKV = nn.ModuleDict(modules={
            key : nn.Linear(in_features=dim, out_features=dim, bias=True) 
            for key in ["Q", "K", "V"]
        })
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.use_att = use_att

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x.shape # (B, N, L)
        Q = self.QKV["Q"](x)
        K = self.QKV["K"](x)
        V = self.QKV["V"](x)
        x, attention_weights = self.multihead_attention(query=Q, key=K, value=V)
        if self.use_att:
            # entropy-based attention refining
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-7), dim=-1) # (B, N)
            attention_scaling = torch.sigmoid(entropy.unsqueeze(-1) - 0.5) * 0.5 + 0.75 # (B, N, 1)
            attention_weights = attention_weights * attention_scaling # (B, N, N)
            attention_weights = torch.softmax(attention_weights, dim=-1) # (B, N, N)
            assert x.dim() == 3 and attention_weights.dim() == 3, f"torch.bmm input must be 3D tensor, got {x.dim()}D and {attention_weights.dim()}D"
            x = torch.bmm(attention_weights, x) # (B, N, L)
        return x

class MLP(nn.Module):
    def __init__(self, in_features : int, out_features : int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Tanh()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        return self.mlp(x)
    
class ResidualBlock(nn.Module):  
    def __init__(self, dim : int) -> None:  
        super().__init__()  
        self.mlp = MLP(in_features=dim, out_features=dim)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:  
        return x + self.mlp(x) 
    
class Classifier(nn.Module):
    def __init__(self, dim : int, num_class : int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            MLP(in_features=dim, out_features=1024),
            ResidualBlock(dim=1024),
            MLP(in_features=1024, out_features=256),
            ResidualBlock(dim=256),
            MLP(in_features=256, out_features=64),
            ResidualBlock(dim=64),
            MLP(in_features=64, out_features=16),
            ResidualBlock(dim=16),
            MLP(in_features=16, out_features=num_class),
        )

    def forward(self, x_p : torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x_p)
        return logits

class DIT_DEP(nn.Module):
    def __init__(self, num_class : int, shape_dict : dict[str, torch.Size], latent_dim : int, use_dit : bool, use_att : bool,
        timesteps : int = 100, schedule : str = "linear"
    ) -> None:
        super().__init__()
        self.use_dit = use_dit
        self.timesteps = timesteps
        # diffusion parameters
        betas = self.__make_beta_schedule__(schedule=schedule, timesteps=timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # cumulative noise decay factor
        self.register_buffer(name="sqrt_alphas_cumprod", tensor=torch.sqrt(alphas_cumprod))
        # cumulative noise growth factor
        self.register_buffer(name="sqrt_one_minus_alphas_cumprod", tensor=torch.sqrt(1.0 - alphas_cumprod))

        # Projector
        self.projector = nn.ModuleDict()
        if Experiment_Config.TS in shape_dict:
            shape = shape_dict[Experiment_Config.TS]
            self.projector[Experiment_Config.TS] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=shape[0]*shape[1], out_features=4096),
                nn.Tanh(),
                nn.Linear(in_features=4096, out_features=latent_dim)
            )
        if Experiment_Config.FC in shape_dict:
            shape = shape_dict[Experiment_Config.FC]
            self.projector[Experiment_Config.FC] = nn.Sequential(
                nn.Linear(in_features=shape[1]*(shape[1]-1)//2, out_features=latent_dim)
            )

        if self.use_dit:
            # Time embedding
            self.time_embedding = Time_Embedding(dim=latent_dim, timesteps=timesteps) 
    
            # Denoising networks
            self.denoising_network = Denoising_Network(dim=latent_dim, use_att=use_att) 

        # Classifier
        self.classifier = Classifier(dim=len(shape_dict)*latent_dim, num_class=num_class)

    def __make_beta_schedule__(self, schedule: str, timesteps: int, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3) -> torch.Tensor:
        """Generates a beta schedule for diffusion models based on the specified type.

        This function creates a sequence of beta values (noise variances) to be used in
        diffusion probabilistic models. The beta schedule controls the noise magnitude
        added at each diffusion timestep, and different schedules affect the diffusion
        process's quality and speed.

        Args:
            schedule (str): The type of beta schedule. Supported values are:
                - "linear": Linearly interpolated betas (in sqrt space, then squared).
                - "cosine": Betas computed to follow a cosine annealing schedule.
                - "sqrt_linear": Betas linearly interpolated between start and end.
                - "sqrt": Betas are the square root of a linear interpolation between start and end.
            timesteps (int): Number of diffusion steps (the schedule length).
            linear_start (float, optional): Start value for linear-based schedules. Default is 1e-4.
            linear_end (float, optional): End value for linear-based schedules. Default is 2e-2.
            cosine_s (float, optional): Small offset for the cosine schedule for numerical stability. Default is 8e-3.

        Returns:
            torch.Tensor: 1-D tensor of shape (timesteps,) containing the beta schedule, 
                with each value between 0 and 1.

        Raises:
            ValueError: If the provided 'schedule' argument is not supported.
        """
        if schedule == "linear":
            # Linear schedule: Compute betas by linearly interpolating between sqrt(linear_start)
            # and sqrt(linear_end), then squaring to obtain a linear schedule in variance.
            # beta_t = (sqrt(linear_start) + t/(T-1)*(sqrt(linear_end) - sqrt(linear_start)))**2 for t in [0, T-1]
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float32) ** 2
        elif schedule == "cosine":
            # Cosine schedule: Calculate an annealed schedule based on an adjusted cosine function,
            # as described in "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
            # 1. Create an array of (timesteps + 1) fractions of time from 0 to 1,
            #    then add offset cosine_s for stability; scale by pi/2 for the cosine.
            # 2. Compute alphas (cumulative product of (1 - beta)) as the squared cosine of the scaled times.
            # 3. Normalize so that alphas[0] == 1, then derive betas via the standard relationship:
            #       beta_t = 1 - alpha_t / alpha_{t-1}
            # 4. Clamp the resulting betas to [0, 0.999] for numerical stability.
            timesteps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps + cosine_s
            alphas = timesteps / (1 + cosine_s) * torch.pi / 2              # Scale fractions to [0, pi/2]
            alphas = torch.cos(alphas).pow(2)                               # Squared cosine for decay profile
            alphas = alphas / alphas[0]                                     # Normalize so alphas[0] == 1
            betas = 1 - alphas[1:] / alphas[:-1]                            # Compute betas from consecutive alphas
            betas = torch.clamp(betas, min=0, max=0.999)                    # Clamp for stability
        elif schedule == "sqrt_linear":
            # Sqrt_linear schedule: Betas increase linearly between linear_start and linear_end.
            # No additional transformation is applied.
            betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float32)
        elif schedule == "sqrt":
            # Sqrt schedule: Calculate betas as the square root of a linearly increasing sequence.
            # This may provide a gentler increase in beta values.
            betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float32) ** 0.5
        else:
            # Raise an error for any unknown schedule type.
            raise ValueError(f"schedule '{schedule}' unknown.")
        # The final betas tensor defines the noise variance at each diffusion timestep.
        return betas

    def __extract_into_tensor__(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:  
        """  
        Extracts values from a tensor `a` based on indices `t` and reshapes the result to match the dimensions of `x_shape`.  

        This function is typically used in diffusion models to extract time-step-dependent values (e.g., alpha or beta values)  
        and broadcast them to match the shape of the input tensor for further computations.  

        Args:  
            a (torch.Tensor): A 1D tensor containing values (e.g., time-step-dependent values like alpha or beta).  
                              The size of `a` should be at least as large as the maximum value in `t`.  
            t (torch.Tensor): A 1D tensor of indices specifying which values to extract from `a`.  
                              The size of `t` is typically equal to the batch size.  
            x_shape (torch.Size): The target shape of the output tensor. The number of dimensions in `x_shape` determines  
                                  how the extracted values are reshaped.  

        Returns:  
            torch.Tensor: A tensor of shape `(t.shape[0], 1, 1, ..., 1)` (same number of dimensions as `x_shape`),  
                          where the first dimension corresponds to the batch size, and all other dimensions are singleton  
                          dimensions (1). This allows the tensor to be broadcasted to match `x_shape`.  

        Example:  
            Suppose `a` is a tensor of time-step-dependent values, `t` is a batch of time-step indices, and `x_shape`  
            is the shape of the input tensor. This function extracts the values from `a` corresponding to the indices in `t`  
            and reshapes them to be broadcast-compatible with `x_shape`.  

            a = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Time-step-dependent values  
            t = torch.tensor([0, 2, 3])            # Batch of time-step indices  
            x_shape = torch.Size([3, 64, 64, 3])   # Target shape (e.g., input tensor shape)  

            result = __extract_into_tensor__(a, t, x_shape)  
            # result will have shape (3, 1, 1, 1), allowing it to be broadcasted to (3, 64, 64, 3).  

        Raises:  
            IndexError: If any value in `t` is out of bounds for the size of `a`.  
        """  
        # Gather values from `a` using indices `t` along the last dimension (-1).  
        # This extracts the values from `a` corresponding to the indices in `t`.  
        gathered = a.gather(-1, t)  
        # Reshape the gathered values to match the target shape.  
        # The first dimension is the batch size (t.shape[0]), and all other dimensions are set to 1.  
        # This ensures the output tensor is broadcast-compatible with `x_shape`.  
        reshaped = gathered.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))  
        return reshaped

    def forward_chain(self, x_0 : torch.Tensor, t : torch.Tensor) -> dict[str, torch.Tensor]:
        """ q_sample, perturb data with noise
        x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        """
        noise = torch.randn_like(input=x_0)
        sqrt_alphas_cumprod_t = self.__extract_into_tensor__(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.__extract_into_tensor__(a=self.sqrt_one_minus_alphas_cumprod, t=t, x_shape=x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def reverse_chain(self, x_t : torch.Tensor, t : torch.Tensor) -> dict[str, torch.Tensor]:
        time_embedding = self.time_embedding(t=t)
        x_p = self.denoising_network(x_t + time_embedding)
        return x_p

    def __get_triu__(self, x : torch.Tensor) -> torch.Tensor:
        """ get the strict upper triangular part of the matrix """
        assert x.dim() == 3 and x.shape[-2] == x.shape[-1], f"Invalid shape: {x.shape}"
        _, matrix_size, _ = x.shape
        mask = torch.triu(input=torch.ones(matrix_size, matrix_size, dtype=torch.bool, device=x.device), diagonal=1)
        return x[:, mask]
    
    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        """
        time series             : (B, num_regions, num_slices)
        functional connectivity : (B, num_regions, num_regions)
        """
        # projected to the same shape: (B, N, latent_dim)
        projected_dict = {}
        if Experiment_Config.TS in input_dict:
            projected_dict[Experiment_Config.TS] = self.projector[Experiment_Config.TS](input_dict[Experiment_Config.TS])
        for key, value in input_dict.items():
            if key == Experiment_Config.FC:
                value = self.__get_triu__(x=value)
            projected_dict[key] = self.projector[key](value)
        tensor = torch.stack(tensors=list(projected_dict.values()), dim=1)

        # diffusion with Transformer
        if self.use_dit:
            # t: time steps in the diffusion process
            # randomly generate a time step t, i.e., directly sample the t-th step of T steps; there is no need to use 'for t in range(T)' to accumulate.
            t = torch.randint(low=0, high=self.timesteps, size=(next(iter(input_dict.values())).shape[0],), dtype=torch.long, device=device) # (batch_size,)
            # forward chain
            x_t = self.forward_chain(x_0=tensor, t=t)
            # reverse chain
            x_p = self.reverse_chain(x_t=x_t, t=t)
        else:
            x_p = tensor

        # classification
        logits = self.classifier(x_p=x_p)

        return Model_Returns(logits=logits, x_proj=tensor, x_pred=x_p)

class Combined_Loss(nn.modules.loss._Loss):
    def __init__(self, cet_weight : float, mse_weight : float, use_aux : bool) -> None:
        super().__init__()
        self.cet_weight = cet_weight
        self.mse_weight = mse_weight
        self.use_aux = use_aux
        self.cet = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, input : torch.Tensor, target : torch.Tensor, x_0 : torch.Tensor, x_p : torch.Tensor) -> None:
        cet_loss = self.cet(input=input, target=target)
        if self.use_aux:
            assert x_0 is not None and x_p is not None
            mse_loss = self.mse(x_0.flatten(start_dim=1), x_p.flatten(start_dim=1))
        else:
            mse_loss = torch.tensor(data=0.0, device=input.device)
        return self.cet_weight * cet_loss + self.mse_weight * mse_loss