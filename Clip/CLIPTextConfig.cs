using System.Text.Json.Serialization;

namespace SD;

public class CLIPTextConfig
{
    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 49408;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 512;

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 2048;

    [JsonPropertyName("projection_dim")]
    public int ProjectionDim { get; set; } = 512;

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; } = 12;

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; } = 8;

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; } = 77;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "quick_gelu";

    [JsonPropertyName("layer_norm_eps")]
    public double LayerNormEps { get; set; } = 1e-5;

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; } = 0.02;

    [JsonPropertyName("initializer_factor")]
    public double InitializerFactor { get; set; } = 1.0;

    [JsonPropertyName("pad_token_id")]
    public int PadTokenId { get; set; } = 1;

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; } = 49406;

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; } = 49407;

    [JsonPropertyName("use_attention_mask")]
    public bool UseAttentionMask { get; set; } = false;

    [JsonPropertyName("dtype")]
    public ScalarType DType { get; set; } = ScalarType.Float32;
}