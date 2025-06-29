import torch
import torch.nn as nn

# from config import Experiment_Config

device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

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

class Denoise_Network(nn.Module):
    def __init__(self, dim : int) -> None:
        super().__init__()
        self.QKV = nn.ModuleDict(modules={key : nn.Linear(in_features=dim, out_features=dim, bias=True) for key in ["Q", "K", "V"]})
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x.shape # (B, N, L)
        return x

class Classifier(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    
    def forward(self, ) -> torch.Tensor:
        pass

class DIT_DEP(nn.Module):
    def __init__(self, shape_dict : dict[str, torch.Size], timesteps : int = 10000, schedule : str = "cosine") -> None:
        super().__init__()
        self.timesteps = timesteps
        # diffusion parameters
        betas = self.__make_beta_schedule__(schedule=schedule, timesteps=timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # cumulative noise decay factor
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        # cumulative noise growth factor
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Time embedding
        self.time_embedding = nn.ModuleDict(modules={
            k : Time_Embedding(dim=v[-1], timesteps=timesteps) 
            for k,v in shape_dict.items()
        })

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

    def forward_chain(self, input_dict : dict[str, torch.Tensor], t : torch.Tensor) -> dict[str, torch.Tensor]:
        """ q_sample, perturb data with noise
        x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        """
        output_dict = {}
        for key, x_0 in input_dict.items():
            noise = torch.randn_like(input=x_0)
            sqrt_alphas_cumprod_t = self.__extract_into_tensor__(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_0.shape)
            sqrt_one_minus_alphas_cumprod_t = self.__extract_into_tensor__(a=self.sqrt_one_minus_alphas_cumprod, t=t, x_shape=x_0.shape)
            x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
            output_dict[key] = x_t
        return output_dict

    def reverse_chain(self, input_dict : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_dict = {}
        for key, value in input_dict.items():
            pass
        return output_dict

    def forward(self, input_dict : dict[str, torch.Tensor]) -> torch.Tensor:
        """
        time series             : (B, num_regions, num_slices)
        functional connectivity : (B, num_regions, num_regions)
        """
        # t: time steps in the diffusion process
        # randomly generate a time step t, i.e., directly sample the t-th step of T steps; there is no need to use 'for t in range(T)' to accumulate.
        t = torch.randint(0, self.timesteps, (next(iter(input_dict.values())).shape[0],), dtype=torch.long, device=device) # (batch_size,)
        
        x_t_dict = self.forward_chain(input_dict=input_dict, t=t)
        time_embedding_dict = self.time_embedding(t=t)
        for key in x_t_dict:
            x_t_dict[key] = x_t_dict[key] + time_embedding_dict[key]
        

        return None