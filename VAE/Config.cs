using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SD
{
    public class Config
    {
        [JsonPropertyName("_class_name")]
        public string ClassName { get; set; } = "AutoencoderKL";

        [JsonPropertyName("_diffusers_version")]
        public string DiffusersVersion { get; set; } = "0.8.0";

        [JsonPropertyName("_name_or_path")]
        public string NameOrPath { get; set; } = "hf-models/stable-diffusion-v2-768x768/vae";

        [JsonPropertyName("act_fn")]
        public string ActivationFunction { get; set; } = "silu";

        [JsonPropertyName("block_out_channels")]
        public int[] BlockOutChannels { get; set; } = { 128, 256, 512, 512 };

        [JsonPropertyName("down_block_types")]
        public string[] DownBlockTypes { get; set; } = { "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D" };

        [JsonPropertyName("in_channels")]
        public int InChannels { get; set; } = 3;

        [JsonPropertyName("latent_channels")]
        public int LatentChannels { get; set; } = 4;

        [JsonPropertyName("layers_per_block")]
        public int LayersPerBlock { get; set; } = 2;

        [JsonPropertyName("norm_num_groups")]
        public int NormNumGroups { get; set; } = 32;

        [JsonPropertyName("out_channels")]
        public int OutChannels { get; set; } = 3;

        [JsonPropertyName("sample_size")]
        public int SampleSize { get; set; } = 768;

        [JsonPropertyName("up_block_types")]
        public string[] UpBlockTypes { get; set; } = { "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D" };

        [JsonPropertyName("scaling_factor")]
        public float ScalingFactor { get; set; } = 0.18215f;

        [JsonPropertyName("latents_mean")]
        public float[]? LatentsMean { get; set; }

        [JsonPropertyName("latents_std")]
        public float[]? LatentsStd { get; set; }

        [JsonPropertyName("force_upcast")]
        public bool ForceUpcast { get; set; } = true;

        public override string ToString()
        {
            return JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
        }
    }
}