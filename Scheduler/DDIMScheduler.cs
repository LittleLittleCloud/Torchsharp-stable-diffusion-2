namespace SD;

public class DDIMSchedulerOutput
{
    /// <param name="prev_sample">`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images</param>
    /// <param name="pred_original_sample">`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images</param>
    public DDIMSchedulerOutput(Tensor prev_sample, Tensor pred_original_sample)
    {
        PrevSample = prev_sample;
        PredOriginalSample = pred_original_sample;
    }

    public Tensor PrevSample { get; set;}
    public Tensor? PredOriginalSample { get; set;}
}

public class DDIMScheduler
{
    public DDIMScheduler(DDIMSchedulerConfig config)
    {
        Config = config;

        if (config.TrainedBetas is not null)
        {
            this.betas = torch.tensor(config.TrainedBetas, dtype: ScalarType.Float32);
        }
        else if (config.BetaSchedule == "linear")
        {
            this.betas = torch.linspace(config.BetaStart, config.BetaEnd, config.NumTrainTimesteps, dtype: ScalarType.Float32);
        }
        else
        {
            throw new ArgumentException("Invalid beta_schedule: " + config.BetaSchedule);
        }

        if (config.RescaleBetasZeroSnr)
        {
            this.betas = Utils.RescaleZeroTerminalSnr(this.betas);
        }

        this.alphas = 1.0f - this.betas;
        this.alphas_cumprod = torch.cumprod(this.alphas, 0);

        // At every step in ddim, we are looking into the previous alphas_cumprod
        // For the final step, there is no previous alphas_cumprod because we are already at 0
        // `set_alpha_to_one` decides whether we set this parameter simply to one or
        // whether we use the final alpha of the "non-previous" one.
        this.final_alpha_cumprod = config.SetAlphaToOne ? torch.tensor(1.0f) : this.alphas_cumprod[0];

        // standard deviation of the initial noise distribution
        this.init_noise_sigma = 1.0f;

        // setable values
        this.num_inference_steps = null;
        var numSteps = Enumerable.Range(0, config.NumTrainTimesteps).Reverse().ToArray();
        this.timesteps = torch.tensor(numSteps, dtype: ScalarType.Int64);
    }

    private Tensor betas;
    private Tensor alphas;
    private Tensor alphas_cumprod;
    private Tensor final_alpha_cumprod;
    private Tensor init_noise_sigma;
    private Tensor timesteps;
    private int? num_inference_steps;
    public DDIMSchedulerConfig Config { get; }
    public Tensor TimeSteps => this.timesteps;
    
    /// <summary>
    /// Ensures interchangeability with schedulers that need to scale the denoising model input depending on the current timestep.
    /// </summary>
    /// <param name="sample">The input sample.</param>
    /// <param name="timestep">The current timestep in the diffusion chain.</param>
    public Tensor ScaleModelInput(Tensor sample, int? timestep = null)
    {
        return sample;
    }

    /// <summary>
    /// Sets the discrete timesteps used for the diffusion chain (to be run before inference).
    /// </summary>
    /// <param name="num_inference_steps">The number of diffusion steps used when generating samples with a pre-trained model.</param>
    public void SetTimesteps(
        int num_inference_steps,
        DeviceType? device = null
    )
    {
        if (num_inference_steps > this.Config.NumTrainTimesteps)
        {
            throw new ArgumentException("num_inference_steps must be less than or equal to num_train_timesteps");
        }

        this.num_inference_steps = num_inference_steps;

        // "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        Tensor timesteps;
        if (this.Config.TimestepSpacing == "linspace")
        {
            timesteps = torch.linspace(0, this.Config.NumTrainTimesteps - 1, num_inference_steps).round().flip().to(ScalarType.Int64);
        }
        else if (this.Config.TimestepSpacing == "leading")
        {
            var step_ratio = this.Config.NumTrainTimesteps / num_inference_steps;
            timesteps = torch.arange(0, this.Config.NumTrainTimesteps, step_ratio).round().to(ScalarType.Int64).flip();
            timesteps += this.Config.StepsOffset;
        }
        else if (this.Config.TimestepSpacing == "trailing")
        {
            var step_ratio = this.Config.NumTrainTimesteps * 1.0f / num_inference_steps;
            timesteps = torch.arange(0, this.Config.NumTrainTimesteps, step_ratio).round().to(ScalarType.Int64).flip();
            timesteps -= 1;
        }
        else
        {
            throw new ArgumentException("Invalid timestep_spacing: " + this.Config.TimestepSpacing);
        }

        if (device is not null)
        {
            timesteps = timesteps.to(device.Value);
        }

        this.timesteps = timesteps;
    }

    public Tensor AddNoise(Tensor original_samples, Tensor noise, Tensor timesteps)
    {
        var alphas_cumprod = this.alphas_cumprod.to(original_samples.device).to(original_samples.dtype);
        timesteps = timesteps.to(original_samples.device);

        var sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps]);
        sqrt_alpha_prod = sqrt_alpha_prod.flatten();
        while(sqrt_alpha_prod.dim() < noise.dim())
        {
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(1);
        }

        var sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod[timesteps]);
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten();
        while(sqrt_one_minus_alpha_prod.dim() < noise.dim())
        {
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(1);
        }

        var noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise;

        return noisy_samples;
    }

    /// <summary>
    /// Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion p
    /// rocess from the learned model outputs (most often the predicted noise).
    /// </summary>
    /// <param name="model_output">The direct output from learned diffusion model.</param>
    /// <param name="timestep">The current discrete timestep in the diffusion chain.</param>
    /// <param name="sample">A current instance of a sample created by the diffusion process.</param>
    /// <param name="eta">The weight of noise for added noise in diffusion step.</param>
    /// <param name="use_clipped_model_output">If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary 
    /// because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no 
    /// clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
    /// `use_clipped_model_output` has no effect.</param>
    /// <param name="variance_noise">Alternative to generating noise with `generator` by directly providing the noise for the variance 
    /// itself. Useful for methods such as [`CycleDiffusion`].</param>
    /// <param name="generator">A random number generator.</param>
    /// <returns></returns>
    public DDIMSchedulerOutput Step(
        Tensor model_output,
        int timestep,
        Tensor sample,
        float eta = 0.0f,
        bool use_clipped_model_output = false,
        Tensor? variance_noise = null,
        Generator? generator = null
    )
    {
        // See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        // Ideally, read DDIM paper in-detail understanding
        // Notation (<variable name> -> <name in paper>
        // - pred_noise_t -> e_theta(x_t, t)
        // - pred_original_sample -> f_theta(x_t, t) or x_0
        // - std_dev_t -> sigma_t
        // - eta -> η
        // - pred_sample_direction -> "direction pointing to x_t"
        // - pred_prev_sample -> "x_t-1"

        if (this.num_inference_steps is null)
        {
            throw new ArgumentException("Set the number of inference steps with `set_timesteps` before running inference.");
        }

        // 1. get previous step value (=t-1)
        var prev_timestep = timestep - this.Config.NumTrainTimesteps / this.num_inference_steps;

        // 2. compute alphas, betas
        var alpha_prod_t = this.alphas_cumprod[timestep];
        var alpha_prod_t_prev = prev_timestep >= 0 ? this.alphas_cumprod[prev_timestep] : this.final_alpha_cumprod;

        var beta_prod_t = 1 - alpha_prod_t;

        Tensor pred_original_sample;
        Tensor pred_epsilon;
        // 3. compute predicted original sample from predicted noise also called
        // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if (this.Config.PredictionType == "epsilon")
        {
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt();
            pred_epsilon = model_output;
        }
        else if (this.Config.PredictionType == "sample")
        {
            pred_original_sample = model_output;
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt();
        }
        else if (this.Config.PredictionType == "v_prediction")
        {
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output;
            pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample;
        }
        else
        {
            throw new ArgumentException("Invalid prediction_type: " + this.Config.PredictionType);
        }

        // 4. Clip or threshold "predicted x_0"
        if (this.Config.Thresholding)
        {
            pred_original_sample = this._ThresholdSample(pred_original_sample);
        }
        else if (this.Config.ClipSample)
        {
            pred_original_sample = torch.clamp(pred_original_sample, min: -this.Config.ClipSampleRange, max: this.Config.ClipSampleRange);
        }

        // 5. compute variance: "sigma_t(η)" -> see formula (16)
        // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        var variance = this._GetVariance(timestep, prev_timestep.Value);
        var std_dev_t = eta * variance.sqrt();

        if (use_clipped_model_output)
        {
            // the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt();
        }

        // 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        var pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t.pow(2)).sqrt() * pred_epsilon;

        // 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        var prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction;

        if (eta > 0)
        {
            if (variance_noise is not null && generator is not null)
            {
                throw new ArgumentException("Only one of variance_noise and generator should be provided.");
            }

            if (variance_noise is null)
            {
                variance_noise = torch.rand(model_output.shape, generator: generator).to(model_output.device).to_type(model_output.dtype);
            }

            variance_noise = variance_noise * std_dev_t;
            prev_sample = prev_sample + variance_noise;
        }


        return new DDIMSchedulerOutput(prev_sample, pred_original_sample);
    }

    public Tensor _GetVariance(int timestep, int prev_timestep)
    {
        var alpha_prod_t = this.alphas_cumprod[timestep];
        var alpha_prod_t_prev = prev_timestep >= 0 ? this.alphas_cumprod[prev_timestep] : this.final_alpha_cumprod;

        var beta_prod_t = 1 - alpha_prod_t;
        var beta_prod_t_prev = 1 - alpha_prod_t_prev;

        var variance = (beta_prod_t_prev / alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev);

        return variance;
    }

    /// <summary>
    /// "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 
    /// (the prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by s.
    /// Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing 
    /// pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    /// photorealism as well as better image-text alignment, especially when using very large guidance weights."
    /// https://arxiv.org/abs/2205.11487
    /// </summary>
    /// <param name="sample"></param>
    /// <returns></returns>
    public Tensor _ThresholdSample(Tensor sample)
    {
        var dtype = sample.dtype;
        var batch_size = sample.shape[0];
        var channels = sample.shape[1];
        var remaining_dims = sample.shape[2..];

        if (dtype != ScalarType.Float32 || dtype != ScalarType.Float64)
        {
            // upcast for quantile calculation, and clamp not implemented for cpu half
            sample = sample.to_type(ScalarType.Float32);
        }

        // Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, -1);

        var abs_sample = torch.abs(sample);

        var s = torch.quantile(abs_sample, this.Config.DynamicThresholdingRatio, dim: 1);
        s = torch.clamp(s, min: 1, max: this.Config.SampleMaxValue);
        s = s.unsqueeze(1); // (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, min: -s, max: s) / s;

        var reshapeShape = new long[] { batch_size, channels };
        reshapeShape = reshapeShape.Concat(remaining_dims).ToArray();
        sample = sample.reshape(reshapeShape);

        sample = sample.to(dtype);

        return sample;
    }
}