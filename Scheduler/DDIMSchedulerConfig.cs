using System.Text.Json.Serialization;

namespace SD;

public class DDIMSchedulerConfig
{
    public DDIMSchedulerConfig(
        int numTrainTimesteps = 1000,
        float betaStart = 0.0001f,
        float betaEnd = 0.02f,
        string betaSchedule = "linear",
        float[]? trainedBetas = null,
        bool clipSample = true,
        bool setAlphaToOne = true,
        int stepsOffset = 0,
        string predictionType = "epsilon",
        bool thresholding = false,
        float dynamicThresholdingRatio = 0.995f,
        float clipSampleRange = 1.0f,
        float sampleMaxValue = 1.0f,
        string timestepSpacing = "leading",
        bool rescaleBetasZeroSnr = false)
    {
        NumTrainTimesteps = numTrainTimesteps;
        BetaStart = betaStart;
        BetaEnd = betaEnd;
        BetaSchedule = betaSchedule;
        TrainedBetas = trainedBetas;
        ClipSample = clipSample;
        SetAlphaToOne = setAlphaToOne;
        StepsOffset = stepsOffset;
        PredictionType = predictionType;
        Thresholding = thresholding;
        DynamicThresholdingRatio = dynamicThresholdingRatio;
        ClipSampleRange = clipSampleRange;
        SampleMaxValue = sampleMaxValue;
        TimestepSpacing = timestepSpacing;
        RescaleBetasZeroSnr = rescaleBetasZeroSnr;
    }

    [JsonPropertyName("num_train_timesteps")]
    public int NumTrainTimesteps { get; set; } = 1000;

    [JsonPropertyName("beta_start")]
    public float BetaStart { get; set; } = 0.0001f;

    [JsonPropertyName("beta_end")]
    public float BetaEnd { get; set; } = 0.02f;

    [JsonPropertyName("beta_schedule")]
    public string BetaSchedule { get; set; } = "linear";

    [JsonPropertyName("trained_betas")]
    public float[]? TrainedBetas { get; set; }

    [JsonPropertyName("clip_sample")]
    public bool ClipSample { get; set; } = true;

    [JsonPropertyName("set_alpha_to_one")]
    public bool SetAlphaToOne { get; set; } = true;

    [JsonPropertyName("steps_offset")]
    public int StepsOffset { get; set; } = 0;

    [JsonPropertyName("prediction_type")]
    public string PredictionType { get; set; } = "epsilon";

    [JsonPropertyName("thresholding")]
    public bool Thresholding { get; set; } = false;

    [JsonPropertyName("dynamic_thresholding_ratio")]
    public float DynamicThresholdingRatio { get; set; } = 0.995f;

    [JsonPropertyName("clip_sample_range")]
    public float ClipSampleRange { get; set; } = 1.0f;

    [JsonPropertyName("sample_max_value")]
    public float SampleMaxValue { get; set; } = 1.0f;

    [JsonPropertyName("timestep_spacing")]
    public string TimestepSpacing { get; set; } = "leading";

    [JsonPropertyName("rescale_betas_zero_snr")]
    public bool RescaleBetasZeroSnr { get; set; } = false;
}