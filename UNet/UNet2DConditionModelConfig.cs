using System.Text.Json.Serialization;

namespace SD;
public class UNet2DConditionModelConfig
{
    [JsonPropertyName("sample_size")]
    public int? SampleSize {get; set;} = null;

    [JsonPropertyName("in_channels")]
    public int InChannels {get; set;} = 4;

    [JsonPropertyName("out_channels")]
    public int OutChannels {get; set;} = 4;

    [JsonPropertyName("center_input_sample")]
    public bool CenterInputSample {get; set;} = false;

    [JsonPropertyName("flip_sin_to_cos")]
    public bool FlipSinToCos {get; set;} = true;

    [JsonPropertyName("freq_shift")]
    public int FreqShift {get; set;} = 0;

    [JsonPropertyName("down_block_types")]
    public string[] DownBlockTypes {get; set;} = new string[] {
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    };

    [JsonPropertyName("mid_block_type")]
    public string MidBlockType {get; set;} = "UNetMidBlock2DCrossAttn";

    [JsonPropertyName("up_block_types")]
    public string[] UpBlockTypes {get; set;} = new string[] {
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    };

    [JsonPropertyName("only_cross_attention")]
    public bool OnlyCrossAttention {get; set;} = false;

    [JsonPropertyName("block_out_channels")]
    public int[] BlockOutChannels {get; set;} = new int[] {320, 640, 1280, 1280};

    [JsonPropertyName("layers_per_block")]
    public int LayersPerBlock {get; set;} = 2;

    [JsonPropertyName("downsample_padding")]
    public int DownsamplePadding {get; set;} = 1;

    [JsonPropertyName("mid_block_scale_factor")]
    public float MidBlockScaleFactor {get; set;} = 1;

    [JsonPropertyName("dropout")]
    public float Dropout {get; set;} = 0.0f;

    [JsonPropertyName("act_fn")]
    public string ActFn {get; set;} = "silu";

    [JsonPropertyName("norm_num_groups")]
    public int? NormNumGroups {get; set;} = 32;

    [JsonPropertyName("norm_eps")]
    public float NormEps {get; set;} = 1e-5f;

    [JsonPropertyName("cross_attention_dim")]
    public int CrossAttentionDim {get; set;} = 1280;

    [JsonPropertyName("transformer_layers_per_block")]
    public int TransformerLayersPerBlock {get; set;} = 1;

    [JsonPropertyName("reverse_transformer_layers_per_block")]
    public int[]? ReverseTransformerLayersPerBlock {get; set;} = null;

    [JsonPropertyName("encoder_hid_dim")]
    public int? EncoderHidDim {get; set;} = null;

    [JsonPropertyName("encoder_hid_dim_type")]
    public string? EncoderHidDimType {get; set;} = null;

    [JsonPropertyName("attention_head_dim")]
    public int[] AttentionHeadDim {get; set;} = [5, 10, 20, 20];

    [JsonPropertyName("num_attention_heads")]
    public int? NumAttentionHeads {get; set;} = null;

    [JsonPropertyName("dual_cross_attention")]
    public bool DualCrossAttention {get; set;} = false;

    [JsonPropertyName("use_linear_projection")]
    public bool UseLinearProjection {get; set;} = false;

    [JsonPropertyName("class_embed_type")]
    public string? ClassEmbedType {get; set;} = null;

    [JsonPropertyName("addition_embed_type")]
    public string? AdditionEmbedType {get; set;} = null;

    [JsonPropertyName("addition_time_embed_dim")]
    public int? AdditionTimeEmbedDim {get; set;} = null;

    [JsonPropertyName("num_class_embeds")]
    public int? NumClassEmbeds {get; set;} = null;

    [JsonPropertyName("upcast_attention")]
    public bool UpcastAttention {get; set;} = false;

    [JsonPropertyName("resnet_time_scale_shift")]
    public string ResnetTimeScaleShift {get; set;} = "default";

    [JsonPropertyName("resnet_skip_time_act")]
    public bool ResnetSkipTimeAct {get; set;} = false;

    [JsonPropertyName("resnet_out_scale_factor")]
    public float ResnetOutScaleFactor {get; set;} = 1.0f;

    [JsonPropertyName("time_embedding_type")]
    public string TimeEmbeddingType {get; set;} = "positional";

    [JsonPropertyName("time_embedding_dim")]
    public int? TimeEmbeddingDim {get; set;} = null;

    [JsonPropertyName("time_embedding_act_fn")]
    public string? TimeEmbeddingActFn {get; set;} = null;

    [JsonPropertyName("timestep_post_act")]
    public string? TimestepPostAct {get; set;} = null;

    [JsonPropertyName("time_cond_proj_dim")]
    public int? TimeCondProjDim {get; set;} = null;

    [JsonPropertyName("conv_in_kernel")]
    public int ConvInKernel {get; set;} = 3;

    [JsonPropertyName("conv_out_kernel")]
    public int ConvOutKernel {get; set;} = 3;

    [JsonPropertyName("projection_class_embeddings_input_dim")]
    public int? ProjectionClassEmbeddingsInputDim {get; set;} = null;

    [JsonPropertyName("attention_type")]
    public string AttentionType {get; set;} = "default";

    [JsonPropertyName("class_embeddings_concat")]
    public bool ClassEmbeddingsConcat {get; set;} = false;

    [JsonPropertyName("mid_block_only_cross_attention")]
    public bool MidBlockOnlyCrossAttention {get; set;} = false;

    [JsonPropertyName("cross_attention_norm")]
    public string? CrossAttentionNorm {get; set;} = null;

    [JsonPropertyName("addition_embed_type_num_heads")]
    public int AdditionEmbedTypeNumHeads {get; set;} = 64;
}
