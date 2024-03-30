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
        this.up_blocks = new ModuleList();

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
    }

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

    private readonly int? sample_size;
    private readonly Conv2d conv_in;
    private Timesteps? time_proj;
    private TimestepEmbedding? time_embedding;
    private Module<Tensor, Tensor>? time_embed_act;
}