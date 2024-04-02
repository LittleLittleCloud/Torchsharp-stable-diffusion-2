using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class UNet2DConditionModelInput
{
    public Tensor Sample {get;}
    public Tensor Timestep {get;}
    public Tensor EncoderHiddenStates {get;}
    public Tensor? ClassLabels {get;}
    public Tensor? TimestepCond {get;}
    public Tensor? AttentionMask {get;}
    public Dictionary<string, object>? CrossAttentionKwargs {get;}
    public Dictionary<string, Tensor>? AddedCondKwargs {get;}
    public Tensor[]? DownBlockAdditionalResiduals {get;}
    public Tensor? MidBlockAditionalResidual {get;}
    public Tensor[]? DownIntrablockAdditionalResiduals {get;}
    public Tensor? EncoderAttentionMask {get;}
}
public class UNet2DConditionModel: Module<UNet2DConditionModelInput, Tensor>
{
    public UNet2DConditionModel(UNet2DConditionModelConfig config)
        : base(nameof(UNet2DConditionModel))
    {
        this.config = config;
        this.sample_size = config.SampleSize;

        if (config.NumAttentionHeads != null)
        {
            throw new ArgumentNullException("NumAttentionHeads can't be null");
        }

        // If `num_attention_heads` is not defined (which is the case for most models)
        // it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        // The reason for this behavior is to correct for incorrectly named variables that were introduced
        // when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        // Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        // which is why we correct for the naming here.

        var num_attention_heads = config.NumAttentionHeads ?? config.AttentionHeadDim;

        // TODO check config

        // input
        var conv_in_padding = (config.ConvInKernel - 1) / 2;
        this.conv_in = Conv2d(
            inputChannel: config.InChannels,
            outputChannel: config.BlockOutChannels[0],
            kernelSize: config.ConvInKernel,
            padding: conv_in_padding);

        // time
        var (time_embed_dim, timestep_input_dim) = this._set_time_proj(
            config.TimeEmbeddingType,
            config.BlockOutChannels,
            config.FlipSinToCos,
            config.FreqShift,
            config.TimeCondProjDim);
        
        this.time_embedding = new TimestepEmbedding(
            in_channels: timestep_input_dim,
            time_embed_dim: time_embed_dim,
            act_fn: config.ActFn,
            post_act_fn: config.TimestepPostAct,
            cond_proj_dim: config.TimeCondProjDim);

        // todo class embedding
        // todo add_embedding

        if (config.TimeEmbeddingActFn is null)
        {
            this.time_embed_act = null;
        }
        else
        {
            this.time_embed_act = Utils.GetActivation(config.TimeEmbeddingActFn);
        }

        this.down_blocks = new ModuleList<Module<DownBlock2DInput, DownBlock2DOutput>>();
        this.up_blocks = new ModuleList<Module<UpBlock2DInput, Tensor>>();

        var only_cross_attention_list = Enumerable.Repeat(config.OnlyCrossAttention, config.DownBlockTypes.Length).ToList();
        var num_attention_heads_list = Enumerable.Repeat(num_attention_heads, config.DownBlockTypes.Length).ToList();
        var attention_head_dim_list = Enumerable.Repeat(config.AttentionHeadDim, config.DownBlockTypes.Length).ToList();
        var cross_attention_dim_list = Enumerable.Repeat(config.CrossAttentionDim, config.DownBlockTypes.Length).ToList();
        var layers_per_block_list = Enumerable.Repeat(config.LayersPerBlock, config.DownBlockTypes.Length).ToList();
        var transformer_layers_per_block_list = Enumerable.Repeat(config.TransformerLayersPerBlock, config.DownBlockTypes.Length).ToList();

        var blocks_time_embed_dim = config.ClassEmbeddingsConcat ? time_embed_dim * 2 : time_embed_dim;

        // down
        var output_channel = config.BlockOutChannels[0];
        for(int i = 0; i != config.DownBlockTypes.Length; ++ i)
        {
            var down_block_type = config.DownBlockTypes[i];
            var input_channel = output_channel;
            output_channel = config.BlockOutChannels[i];
            var is_final_block = i == config.DownBlockTypes.Length - 1;

            var down_block = Utils.GetDownBlock(
                down_block_type: down_block_type,
                num_layers: layers_per_block_list[i],
                transformer_layers_per_block: transformer_layers_per_block_list[i],
                in_channels: input_channel,
                out_channels: output_channel,
                temb_channels: blocks_time_embed_dim,
                add_downsample: !is_final_block,
                resnet_eps: config.NormEps,
                resnet_act_fn: config.ActFn,
                resnet_groups: config.NormNumGroups,
                cross_attention_dim: cross_attention_dim_list[i],
                num_attention_heads: num_attention_heads_list[i],
                downsample_padding: config.DownsamplePadding,
                dual_cross_attention: config.DualCrossAttention,
                use_linear_projection: config.UseLinearProjection,
                only_cross_attention: only_cross_attention_list[i],
                upcast_attention: config.UpcastAttention,
                resnet_time_scale_shift: config.ResnetTimeScaleShift,
                attention_type: config.AttentionType,
                resnet_skip_time_act: config.ResnetSkipTimeAct,
                resnet_out_scale_factor: config.ResnetOutScaleFactor,
                cross_attention_norm: config.CrossAttentionNorm,
                attention_head_dim: attention_head_dim_list[i],
                dropout: config.Dropout);

            this.down_blocks.Add(down_block);
        }

        // mid
        this.mid_block = Utils.GetMidBlock(
            mid_block_type: config.MidBlockType,
            temb_channels: blocks_time_embed_dim,
            in_channels: config.BlockOutChannels[^1],
            resnet_eps: config.NormEps,
            resnet_act_fn: config.ActFn,
            resnet_groups: config.NormNumGroups ?? throw new ArgumentNullException("NormNumGroups can't be null"),
            output_scale_factor: config.MidBlockScaleFactor,
            transformer_layers_per_block: transformer_layers_per_block_list[^1],
            num_attention_heads: num_attention_heads_list[^1],
            cross_attention_dim: cross_attention_dim_list[^1],
            dual_cross_attention: config.DualCrossAttention,
            use_linear_projection: config.UseLinearProjection,
            mid_block_only_cross_attention: config.MidBlockOnlyCrossAttention,
            upcast_attention: config.UpcastAttention,
            resnet_time_scale_shift: config.ResnetTimeScaleShift,
            attention_type: config.AttentionType,
            resnet_skip_time_act: config.ResnetSkipTimeAct,
            cross_attention_norm: config.CrossAttentionNorm,
            attention_head_dim: attention_head_dim_list[^1],
            dropout: config.Dropout);

        this.num_upsamplers = 0;

        // up
        var reversed_block_out_channels = config.BlockOutChannels.Reverse().ToArray();
        var reversed_num_attention_heads = num_attention_heads_list.ToArray().Reverse().ToArray();
        var reversed_layers_per_block = layers_per_block_list.ToArray().Reverse().ToArray();
        var reversed_cross_attention_dim = cross_attention_dim_list.ToArray().Reverse().ToArray();
        var reversed_transformer_layers_per_block = config.ReverseTransformerLayersPerBlock ?? transformer_layers_per_block_list.ToArray().Reverse().ToArray();

        only_cross_attention_list = only_cross_attention_list.ToArray().Reverse().ToList();
        output_channel = reversed_block_out_channels[0];
        for(int i = 0; i != config.UpBlockTypes.Length; ++i)
        {
            var is_final_block = i == config.UpBlockTypes.Length - 1;
            var prev_output_channel = output_channel;
            output_channel = reversed_block_out_channels[i];
            var input_channel = reversed_block_out_channels[Math.Min(i + 1, reversed_block_out_channels.Length - 1)];

            // add upsample block for all BUT final layer
            bool add_upsample = !is_final_block;
            if (!is_final_block)
            {
                this.num_upsamplers += 1;
            }

            var up_block = Utils.GetUpBlock(
                up_block_type: config.UpBlockTypes[i],
                num_layers: reversed_layers_per_block[i] + 1,
                transformer_layers_per_block: reversed_transformer_layers_per_block[i],
                in_channels: input_channel,
                out_channels: output_channel,
                prev_output_channel: prev_output_channel,
                temb_channels: blocks_time_embed_dim,
                add_upsample: add_upsample,
                resnet_eps: config.NormEps,
                resnet_act_fn: config.ActFn,
                resolution_idx: i,
                resnet_groups: config.NormNumGroups ?? throw new ArgumentNullException("NormNumGroups can't be null"),
                cross_attention_dim: reversed_cross_attention_dim[i],
                num_attention_heads: reversed_num_attention_heads[i],
                dual_cross_attention: config.DualCrossAttention,
                use_linear_projection: config.UseLinearProjection,
                only_cross_attention: only_cross_attention_list[i],
                upcast_attention: config.UpcastAttention,
                resnet_time_scale_shift: config.ResnetTimeScaleShift,
                attention_type: config.AttentionType,
                resnet_skip_time_act: config.ResnetSkipTimeAct,
                resnet_out_scale_factor: config.ResnetOutScaleFactor,
                cross_attention_norm: config.CrossAttentionNorm,
                attention_head_dim: attention_head_dim_list[i],
                dropout: config.Dropout);
            
            this.up_blocks.Add(up_block);
            prev_output_channel = output_channel;
        }

        // out
        if (config.NormNumGroups is not null)
        {
            this.conv_norm_out = GroupNorm(
                num_channels: config.BlockOutChannels[0],
                num_groups: config.NormNumGroups.Value,
                eps: config.NormEps);
            this.conv_act = Utils.GetActivation(config.ActFn);
        }

        var conv_out_padding = (config.ConvOutKernel - 1) / 2;
        this.conv_out = Conv2d(
            inputChannel: config.BlockOutChannels[0],
            outputChannel: config.OutChannels,
            kernelSize: config.ConvOutKernel,
            padding: conv_out_padding);
    }
    private readonly UNet2DConditionModelConfig config;
    private readonly int? sample_size;
    private readonly Conv2d conv_in;
    private Timesteps time_proj;
    private TimestepEmbedding time_embedding;
    private Module<Tensor, Tensor>? time_embed_act;
    private ModuleList<Module<DownBlock2DInput, DownBlock2DOutput>> down_blocks;
    private Module<UNetMidBlock2DInput, Tensor> mid_block;
    private int num_upsamplers;
    private ModuleList<Module<UpBlock2DInput, Tensor>> up_blocks;
    private Module<Tensor, Tensor>? conv_norm_out;
    private Module<Tensor, Tensor>? conv_act;
    private Conv2d conv_out;

    private (int, int) _set_time_proj(
        string time_embedding_type,
        int[] block_out_channels,
        bool flip_sin_to_cos,
        float freq_shift,
        int? time_embedding_dim)
    {
        if (time_embedding_type != "positional")
        {
            throw new ArgumentException("Only positional time embeddings are supported");
        }

        var time_embed_dim = time_embedding_dim ?? block_out_channels[0] * 4;
        var timestep_input_dim = block_out_channels[0];
        this.time_proj = new Timesteps(timestep_input_dim, flip_sin_to_cos, freq_shift);

        return (time_embed_dim, timestep_input_dim);
    }

    private Tensor get_time_embed(
        Tensor sample,
        Tensor timestep)
    {
        var device = sample.device;
        if (timestep.shape.Length == 0)
        {
            timestep = timestep.unsqueeze(0).to(device);
        }

        timestep = timestep.expand(sample.shape[0]);
        var time_embed = this.time_proj.forward(timestep);
        time_embed = time_embed.to_type(sample.dtype);

        return time_embed;
    }

    private Tensor process_encoder_hidden_states(
        Tensor encoder_hidden_states,
        IDictionary<string, object>? added_cond_kwargs)
    {
        // encoder_hid_proj is null

        return encoder_hidden_states;
    }

    public override Tensor forward(UNet2DConditionModelInput input)
    {
        // By default samples have to be AT least a multiple of the overall upsampling factor.
        // The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        // However, the upsampling interpolation output size can be forced to fit any upsampling size
        // on the fly if necessary.
        var default_overall_up_factor = Math.Pow(2, this.num_upsamplers);

        var forward_upsample_size = false;
        int? upsample_size = null;
        var sample = input.Sample;
        foreach(var dim in input.Sample.shape[2..])
        {
            if (dim % default_overall_up_factor != 0)
            {
                forward_upsample_size = true;
                break;
            }
        }

        // ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        // expects mask of shape:
        //   [batch, key_tokens]
        // adds singleton query_tokens dimension:
        //   [batch,                    1, key_tokens]
        // this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        //   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        //   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        var attention_mask = input.AttentionMask;
        if (attention_mask is not null)
        {
            // assume that mask is expressed as:
            //   (1 = keep,      0 = discard)
            // convert mask into a bias that can be added to attention scores:
            //       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0f;
            attention_mask = attention_mask.unsqueeze(1);
        }

        // convert encoder_attention_mask to a bias the same way we do for attention_mask
        var encoder_attention_mask = input.EncoderAttentionMask;
        if (encoder_attention_mask is not null)
        {
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0f;
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1);
        }

        // 0, center input if necessary
        if (this.config.CenterInputSample)
        {
            sample = 2 * sample - 1.0;
        }

        // 1. time
        var t_emb = this.get_time_embed(sample, input.Timestep);
        var emb = this.time_embedding.forward(t_emb, input.TimestepCond);

        // class emb is null

        // aug_emb is null
        if (this.time_embed_act is not null)
        {
            emb = this.time_embed_act.forward(emb);
        }

        var encoder_hidden_states = input.EncoderHiddenStates;
        encoder_hidden_states = this.process_encoder_hidden_states(encoder_hidden_states, input.AddedCondKwargs?.ToDictionary(x => x.Key, x => (object)x.Value));

        // 2. pre-process
        sample = this.conv_in.forward(sample);

        // 2.5 GLIGEN position net
        if (input.CrossAttentionKwargs?.TryGetValue("gligen", out var value) is true)
        {
            throw new NotImplementedException("GLIGEN not implemented");
        }

        // 3 down
        var is_controlnet = false;
        var is_adapter = false;

        var down_block_res_samples = new List<Tensor>();
        down_block_res_samples.Add(sample);
        foreach(var downsample_block in this.down_blocks)
        {
            var downBlockInput = new DownBlock2DInput(
                hiddenStates: sample,
                temb: t_emb,
                encoderHiddenStates: encoder_hidden_states,
                attentionMask: attention_mask,
                encoderAttentionMask: encoder_attention_mask,
                additionalResiduals: null);

            var downBlockOutput = downsample_block.forward(downBlockInput);
            sample = downBlockOutput.HiddenStates;
            var resSamples = downBlockOutput.OutputStates;
            down_block_res_samples.AddRange(resSamples ?? []);
        }

        // 4 mid
        if (this.mid_block is not null)
        {
            var midBlockInput = new UNetMidBlock2DInput(
                hiddenStates: sample,
                temb: t_emb,
                encoderHiddenStates: encoder_hidden_states,
                attentionMask: attention_mask,
                crossAttentionKwargs: input.CrossAttentionKwargs,
                encoderAttentionMask: encoder_attention_mask);

            sample = this.mid_block.forward(midBlockInput);
        }

        // 5 up
        for(int i = 0; i != this.up_blocks.Count; ++i)
        {
            var is_final_block = i == this.up_blocks.Count - 1;
            var upsample_block = this.up_blocks[i];
            var resnets = upsample_block switch
            {
                CrossAttnUpBlock2D crossAttnUpBlock2D => crossAttnUpBlock2D.Resnets,
                UpBlock2D upBlock2D => upBlock2D.Resnets,
                _ => throw new NotImplementedException()
            };
            var resnets_count = resnets.Count;
            var res_samples = down_block_res_samples.ToArray()[(^resnets_count) ..];
            down_block_res_samples = down_block_res_samples.ToArray()[..^resnets_count].ToList();

            // if we have not reached the final block and need to forward the
            // upsample size, we do it here
            if (!is_final_block && forward_upsample_size)
            {
                upsample_size = down_block_res_samples[-1].IntShape()[2];
            }

            var upBlockInput = new UpBlock2DInput(
                hiddenStates: sample,
                resHiddenStatesTuple: res_samples,
                temb: t_emb,
                encoderHiddenStates: encoder_hidden_states,
                crossAttentionKwargs: input.CrossAttentionKwargs,
                upsampleSize: upsample_size,
                attentionMask: attention_mask,
                encoderAttentionMask: encoder_attention_mask);

            sample = upsample_block.forward(upBlockInput);
        }

        // 6. post-process
        if (this.conv_norm_out is not null && this.conv_act is not null)
        {
            sample = this.conv_norm_out.forward(sample);
            sample = this.conv_act.forward(sample);
        }

        sample = this.conv_out.forward(sample);

        return sample;
    }   
}