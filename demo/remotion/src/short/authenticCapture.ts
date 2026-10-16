export const AUTHENTIC_CAPTURE = {
  jobId: "393729",
  model: "Qwen/Qwen3-8B",
  normalStartupSeconds: 107.1,
  speedlmStartupSeconds: 107.1,
  request: "Reply exactly: SpeedLM serves the same vLLM API",
  response: "SpeedLM serves the same vLLM API",
  promptTokens: 30,
  completionTokens: 10,
  idle: {
    quiescingSeconds: 8.0,
    extractingSeconds: 18.1,
    trainingSeconds: 218.7,
    firstOptimizerSeconds: 335.2,
    firstLoss: "10.648",
  },
} as const;
