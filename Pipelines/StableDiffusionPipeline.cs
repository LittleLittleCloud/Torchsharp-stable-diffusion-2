namespace SD;
public class StableDiffusionPipelineOutput
{
    public StableDiffusionPipelineOutput(Tensor images)
    {
        Images = images;
    }

    /// <summary>
    /// The generated images. size (batch_size, ...).
    /// </summary>
    public Tensor Images { get; }
}
public class StableDiffusionPipeline
{
    private readonly int vae_scale_factor;
    private DeviceType device = DeviceType.CPU;

    public StableDiffusionPipeline(
        AutoencoderKL vae,
        CLIPTextModel text_encoder,
        BPETokenizer tokenizer,
        UNet2DConditionModel unet,
        DDIMScheduler scheduler) // todo: safety checker, feature extractor and image encoder
    {
        this.vae = vae;
        this.text_encoder = text_encoder;
        this.tokenizer = tokenizer;
        this.unet = unet;
        this.scheduler = scheduler;

        this.vae_scale_factor = Convert.ToInt32(Math.Pow(2, this.vae.Config.BlockOutChannels.Length - 1));
    }

    public void To(DeviceType device)
    {
        if (device != this.device)
        {
            this.device = device;
            this.text_encoder.to(device);
            this.vae.to(device);
            this.unet.to(device);
        }
    }

    public AutoencoderKL vae { get; }
    public CLIPTextModel text_encoder { get; }
    public BPETokenizer tokenizer { get; }
    public UNet2DConditionModel unet { get; }
    public DDIMScheduler scheduler { get; }

    /// <summary>
    /// Run the stable diffusion pipeline.
    /// </summary>
    /// <param name="prompt">he prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`</param>
    /// <param name="height">The height in pixels of the generated image. defalt to unet.sample_size * vae_scale_factor</param>
    /// <param name="width">The width in pixels of the generated image. defalt to unet.sample_size * vae_scale_factor</param>
    /// <param name="num_inference_steps">The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.</param>
    /// <param name="timesteps">Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument 
    /// in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is 
    /// passed will be used. Must be in descending order.</param>
    /// <param name="guidance_scale">A higher guidance scale value encourages the model to generate images closely linked to the text
    /// `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.</param>
    /// <param name="negative_prompt">The prompt or prompts to guide what to not include in image generation. Ignored when not using guidance (`guidance_scale < 1`).</param>
    /// <param name="num_images_per_prompt">The number of images to generate per prompt.</param>
    /// <param name="eta">Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies 
    /// to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.</param>
    /// <param name="generator"> A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.</param>
    /// <param name="latents">Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
    /// generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
    /// tensor is generated by sampling using the supplied random `generator`.</param>
    /// <param name="prompt_embeds">Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not 
    /// provided, text embeddings are generated from the `prompt` input argument.</param>
    /// <param name="negative_prompt_embeds">Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting).
    /// If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.</param>
    /// <param name="guidance_rescale">Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    /// Guidance rescale factor should fix overexposure when using zero terminal SNR.</param>
    /// <returns></returns>
    public StableDiffusionPipelineOutput Run(
        string? prompt = null,
        int? height = null,
        int? width = null,
        int num_inference_steps = 50,
        int[]? timesteps = null,
        float guidance_scale = 7.5f,
        string? negative_prompt = null,
        int num_images_per_prompt = 1,
        float eta = 0.0f,
        Generator? generator = null,
        Tensor? latents = null,
        Tensor? prompt_embeds = null,
        Tensor? negative_prompt_embeds = null,
        float guidance_rescale = 0.0f)
    {
        using var _ = torch.no_grad();
        height = height ?? unet.Config.SampleSize * this.vae_scale_factor;
        width = width ?? unet.Config.SampleSize * this.vae_scale_factor;
        var do_classifier_free_guidance = guidance_scale > 1.0f && this.unet.Config.TimeCondProjDim is null;

        if (prompt is not null && prompt_embeds is not null)
        {
            throw new ArgumentException("Only one of `prompt` or `prompt_embeds` should be passed.");
        }

        if (negative_prompt is not null && negative_prompt_embeds is not null)
        {
            throw new ArgumentException("Only one of `negative_prompt` or `negative_prompt_embeds` should be passed.");
        }
        // todo
        // deal with lora

        int batch_size = 1;
        if (prompt_embeds is not null)
        {
            batch_size = prompt_embeds.IntShape()[0];
        }

        prompt_embeds = this.EncodePrompt(batch_size, prompt, prompt_embeds, num_images_per_prompt);
        if (do_classifier_free_guidance)
        {
            if (negative_prompt is null && negative_prompt_embeds is null)
            {
                negative_prompt = "";
            }

            negative_prompt_embeds = this.EncodePrompt(batch_size, negative_prompt, negative_prompt_embeds, num_images_per_prompt);

            // For classifier free guidance, we need to do two forward passes.
            // Here we concatenate the unconditional and text embeddings into a single batch
            // to avoid doing two forward passes

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]);
        }

        // prepare timesteps
        (var time_steps_tensor, num_inference_steps) = this.RetireveTimesteps(num_inference_steps, timesteps);

        // prepare latent variables
        var num_channels_latents = this.unet.Config.InChannels;
        latents = this.PrepareLatents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width!.Value,
            height!.Value,
            device,
            generator: generator,
            latents: latents);
        
        // denosing loop
        for(int i = 0; i!= num_inference_steps; i++)
        {
            var step = (int)time_steps_tensor[i].ToInt64();
            Console.WriteLine($"Step {step}");
            // expand the latents if we are doing classifier free guidance
            var latent_model_input = !do_classifier_free_guidance ? latents : torch.cat([latents, latents], 0);
            latent_model_input = this.scheduler.ScaleModelInput(latent_model_input, step);

            latent_model_input.Peek("latent_model_input");
            prompt_embeds.Peek("prompt_embeds");
            // predict noise residual
            Tensor noise_pred;
            using (var __ = NewDisposeScope())
            {
                var unetInput = new UNet2DConditionModelInput(
                    sample: latent_model_input,
                    timestep: time_steps_tensor[i],
                    encoderHiddenStates: prompt_embeds);
                noise_pred = this.unet.forward(unetInput).MoveToOuterDisposeScope();
            }

            if (do_classifier_free_guidance)
            {
                var chunk = noise_pred.chunk(2, 0);
                var noise_pred_uncond = chunk[0];
                var noise_pred_text = chunk[1];

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond);
            }

            // compute the previous noisy sample x_t -> x_{t-1}
            latents = this.scheduler.Step(
                noise_pred,
                step,
                latents).PrevSample;
        }

        // decode to image tensor
        var image_tensor = this.vae.decode(latents / this.vae.Config.ScalingFactor);
        return new StableDiffusionPipelineOutput(image_tensor);
    }

    public Tensor PrepareLatents(
        int batch_size,
        int num_channels_latents,
        int width,
        int height,
        DeviceType device,
        ScalarType dtype = ScalarType.Float32,
        Generator? generator = null,
        Tensor? latents = null)
    {
        long[] shape = [batch_size, num_channels_latents, height / this.vae_scale_factor, width / this.vae_scale_factor];
        Console.WriteLine($"Height: {height}, Width: {width}, vae_scale_factor: {this.vae_scale_factor}");
        if (latents is null)
        {
            latents = torch.randn(shape, dtype: dtype, generator: generator).to(device);
        }
        else
        {
            latents = latents.to(dtype).to(device);
        }

        latents.Peek("latents");
        latents = latents * this.scheduler.InitNoiseSigma;
        latents.Peek("latents");
        Console.WriteLine($"init_noise_sigma: {this.scheduler.InitNoiseSigma}");

        return latents;
    }

    /// <summary>
    /// Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call
    /// </summary>
    /// <param name="num_inference_steps">The number of diffusion steps used when generating samples with a pre-trained model.
    /// If used, `timesteps` must be `None`.</param>
    /// <param name="timesteps">Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default 
    /// timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps` 
    /// must be `None`</param>
    /// <returns>A tuple where the first element is the timestep schedule from the scheduler and the
    /// second element is the number of inference steps.</returns>
    public (Tensor, int) RetireveTimesteps(
        int? num_inference_steps = null,
        int[]? timesteps = null)
    {
        if (num_inference_steps is not null && timesteps is not null)
        {
            throw new ArgumentException("Only one of `num_inference_steps` or `timesteps` should be passed.");
        }

        if (num_inference_steps is null && timesteps is null)
        {
            throw new ArgumentException("Either `num_inference_steps` or `timesteps` must be passed.");
        }

        if (num_inference_steps is not null)
        {
            this.scheduler.SetTimesteps(num_inference_steps.Value);

            return (this.scheduler.TimeSteps, this.scheduler.TimeSteps.IntShape()[0]);
        }
        else
        {
            this.scheduler.SetTimesteps(timesteps: timesteps);
            return (this.scheduler.TimeSteps, timesteps!.Length);
        }
    }

    public Tensor EncodePrompt(
        int batch_size,
        string? prompt = null,
        Tensor? prompt_embeds = null,
        int num_images_per_prompt = 1)
    {
        if (prompt is null && prompt_embeds is null)
        {
            throw new ArgumentException("Either `prompt` or `prompt_embeds` must be passed.");
        }

        if (prompt is not null && prompt_embeds is not null)
        {
            throw new ArgumentException("Only one of `prompt` or `prompt_embeds` should be passed.");
        }

        if (prompt is string)
        {
            // todo
            // enable attention_mask in tokenizer

            var text_inputs_id = this.tokenizer.Encode(prompt, true, true, padding: "max_length", maxLength: tokenizer.ModelMaxLength);
            var id_tensor = torch.tensor(text_inputs_id, dtype: ScalarType.Int64).reshape(1, -1).to(this.device);
            var output = this.text_encoder.forward(id_tensor, attention_mask: null);
            prompt_embeds = output.LastHiddenState;
        }

        var seql_len = prompt_embeds!.IntShape()[1];
        // duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds!.repeat([1, num_images_per_prompt, 1]);
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seql_len, -1);

        return prompt_embeds;
    }
}